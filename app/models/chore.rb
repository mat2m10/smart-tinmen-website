# A household task. One-off tasks are closed when done; repeating tasks move
# their due date forward instead.
class Chore < ApplicationRecord
  # Suggestions only: area and room are free text, so each household keeps
  # its own list in the database (and nothing personal lives in this code).
  DEFAULT_AREAS = [ "Cleaning", "Laundry", "Food", "Kids", "Admin", "Family & friends", "Home & repairs" ].freeze
  REPEAT_UNITS = %w[none day weekday week month year].freeze
  EFFORTS = { 1 => "Quick", 2 => "Medium", 3 => "Big" }.freeze
  # Turns "Empty the dishwasher" into "emptied the dishwasher" for "X did it"
  # notifications. Titles starting with another word fall back to quoting.
  PAST_TENSE = {
    "bring" => "brought", "buy" => "bought", "do" => "did", "feed" => "fed", "find" => "found", "get" => "got",
    "give" => "gave", "go" => "went", "hang" => "hung", "make" => "made", "mop" => "mopped", "pay" => "paid",
    "plan" => "planned", "put" => "put", "run" => "ran", "sell" => "sold", "send" => "sent", "set" => "set",
    "shop" => "shopped", "sweep" => "swept", "take" => "took", "throw" => "threw", "write" => "wrote"
  }.freeze
  REGULAR_VERBS = %w[
    apply book call cancel change check clean clear cook dust email empty fix fold iron order organise organize
    pick prepare register renew repair replace schedule sort tidy vacuum wash water
  ].freeze

  belongs_to :assignee, class_name: "User", optional: true
  belongs_to :created_by, class_name: "User", optional: true
  has_many :completions, dependent: :destroy
  # "notes" is the chore's own text field; the thread of Note records is "remarks".
  has_many :remarks, class_name: "Note", dependent: :nullify
  # Remarks on a personal chore would otherwise become loose notes everyone sees.
  before_destroy -> { remarks.destroy_all if personal }, prepend: true

  validates :title, presence: true, length: { maximum: 200 }
  validates :due_on, presence: true
  validates :area, presence: true, length: { maximum: 40 }
  validates :room, length: { maximum: 40 }
  validates :repeat_unit, inclusion: { in: REPEAT_UNITS }
  validates :repeat_every, numericality: { only_integer: true, in: 1..365 }
  validates :effort, inclusion: { in: EFFORTS.keys }
  validates :assignee, presence: true, unless: -> { assignee_id.nil? }
  # A personal chore belongs to one person and only they see it.
  validates :assignee, presence: { message: "is needed for a personal task" }, if: :personal
  # Optional time of day ("14:30") for appointments; without it a chore is all-day.
  validates :due_time, format: { with: /\A([01]\d|2[0-3]):[0-5]\d\z/ }, allow_nil: true

  # A weekday chore never falls on a weekend, also not the first time.
  before_validation -> { self.due_on = due_on.next_day while due_on.on_weekend? }, if: -> { repeat_unit == "weekday" && due_on }

  normalizes :due_time, with: ->(value) { (match = value.strip.match(/\A(\d{1,2}):(\d{2})(:\d{2})?\z/)) ? format("%02d:%s", match[1].to_i, match[2]) : value.strip.presence }

  # Timed chores are appointments and get an event in the shared Google Calendar.
  CALENDAR_FIELDS = %w[title notes area room due_on due_time assignee_id personal].freeze
  after_commit :sync_calendar, on: %i[create update], if: -> { (due_time || due_time_before_last_save) && saved_changes.keys.intersect?(CALENDAR_FIELDS) }
  # A second after_commit naming the same method would replace the first.
  after_destroy_commit -> { HouseholdCalendarJob.perform_later(id) }, if: -> { due_time }

  scope :pending, -> { where(done_at: nil) }
  scope :finished, -> { where.not(done_at: nil) }
  scope :shared, -> { where(personal: false) }
  scope :visible_to, ->(user) { shared.or(where(assignee: user)) }

  def self.area_options
    (DEFAULT_AREAS + distinct.pluck(:area)).uniq.sort
  end

  def self.room_options
    where.not(room: [ nil, "" ]).distinct.order(:room).pluck(:room)
  end

  # Who else hears about this chore: nobody when it is personal.
  def audience(except:)
    personal ? User.none : User.where.not(id: except)
  end

  def repeating?
    repeat_unit != "none"
  end

  def complete!(user, on: Date.current)
    transaction do
      completions.create!(user: user, completed_on: on, effort: effort, planning: planning, previous_due_on: due_on)
      repeating? ? update!(due_on: next_due_after(on)) : update!(done_at: Time.current)
    end
  end

  # Takes a tick back. Only the latest one puts the chore back as it was
  # (also over a date edited by hand since); an older one just leaves the
  # history. Ticks from before previous_due_on existed fall back to the day done.
  def undo!(completion)
    transaction do
      if completion == completions.order(:created_at, :id).last
        update!(done_at: nil, due_on: completion.previous_due_on || (repeating? ? completion.completed_on : due_on))
      end
      completion.destroy!
    end
  end

  # Fixed schedules keep their calendar rhythm (rent on the 25th); the others
  # restart from the day the chore was done (clean the shower a week later).
  def next_due_after(on)
    return advance(on) unless fixed_schedule

    date = advance(due_on)
    date = advance(date) while date <= on
    date
  end

  # The days in range this chore falls on: its due date, then (for repeating
  # chores) the dates after it. Those later dates are a forecast: unless the
  # schedule is fixed, they move when the chore is done early or late.
  def occurrences(range)
    dates = []
    date = due_on
    while date <= range.end
      dates << date if range.cover?(date)
      next_date = advance(date)
      break if next_date <= date # one-off chores, and a guard against bad data

      date = next_date
    end
    dates
  end

  def bucket(today = Date.current)
    if due_on < today then :overdue
    elsif due_on == today then :today
    elsif due_on <= today + 6 then :week
    else :later
    end
  end

  def done_sentence(name)
    verb, rest = title.strip.split(/\s+/, 2)
    past = PAST_TENSE[verb.downcase] || (regular_past(verb.downcase) if REGULAR_VERBS.include?(verb.downcase))
    past ? [ name, past, rest ].compact.join(" ") : "#{name} finished “#{title.strip}”"
  end

  def repeat_label
    return "Once" unless repeating?

    repeat_every == 1 ? "Every #{repeat_unit}" : "Every #{repeat_every} #{repeat_unit.pluralize}"
  end

  def effort_label
    EFFORTS.fetch(effort, effort.to_s)
  end

  def last_done_on
    completions.maximum(:completed_on)
  end

  def as_api_json
    {
      id: id, title: title, notes: notes, area: area, room: room, due_on: due_on, due_time: due_time,
      repeat_unit: repeat_unit, repeat_every: repeat_every, fixed_schedule: fixed_schedule,
      effort: effort, planning: planning, personal: personal, assignee_email: assignee&.email,
      done_at: done_at, last_done_on: last_done_on, updated_at: updated_at
    }
  end

  private

  def sync_calendar
    HouseholdCalendarJob.perform_later(id)
  end

  def regular_past(verb)
    if verb.end_with?("e") then "#{verb}d"
    elsif verb.match?(/[^aeiou]y\z/) then "#{verb.delete_suffix("y")}ied"
    else "#{verb}ed"
    end
  end

  def advance(date)
    case repeat_unit
    when "day" then date.advance(days: repeat_every)
    # Monday to Friday only, e.g. daycare drop-off: Friday moves on to Monday.
    when "weekday" then repeat_every.times.reduce(date) { |day, _| day = day.next_day; day = day.next_day while day.on_weekend?; day }
    when "week" then date.advance(weeks: repeat_every)
    when "month" then date.advance(months: repeat_every)
    when "year" then date.advance(years: repeat_every)
    else date
    end
  end
end
