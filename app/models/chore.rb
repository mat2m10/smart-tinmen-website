# A household task. One-off tasks are closed when done; repeating tasks move
# their due date forward instead.
class Chore < ApplicationRecord
  # Suggestions only: area and room are free text, so each household keeps
  # its own list in the database (and nothing personal lives in this code).
  DEFAULT_AREAS = [ "Cleaning", "Laundry", "Food", "Kids", "Admin", "Family & friends", "Home & repairs" ].freeze
  REPEAT_UNITS = %w[none day week month year].freeze
  EFFORTS = { 1 => "Quick", 2 => "Medium", 3 => "Big" }.freeze

  belongs_to :assignee, class_name: "User", optional: true
  belongs_to :created_by, class_name: "User", optional: true
  has_many :completions, dependent: :destroy
  # "notes" is the chore's own text field; the thread of Note records is "remarks".
  has_many :remarks, class_name: "Note", dependent: :nullify

  validates :title, presence: true, length: { maximum: 200 }
  validates :due_on, presence: true
  validates :area, presence: true, length: { maximum: 40 }
  validates :room, length: { maximum: 40 }
  validates :repeat_unit, inclusion: { in: REPEAT_UNITS }
  validates :repeat_every, numericality: { only_integer: true, in: 1..365 }
  validates :effort, inclusion: { in: EFFORTS.keys }
  validates :assignee, presence: true, unless: -> { assignee_id.nil? }
  # Optional time of day ("14:30") for appointments; without it a chore is all-day.
  validates :due_time, format: { with: /\A([01]\d|2[0-3]):[0-5]\d\z/ }, allow_nil: true

  normalizes :due_time, with: ->(value) { (match = value.strip.match(/\A(\d{1,2}):(\d{2})/)) ? format("%02d:%s", match[1].to_i, match[2]) : value.strip.presence }

  scope :pending, -> { where(done_at: nil) }
  scope :finished, -> { where.not(done_at: nil) }

  def self.area_options
    (DEFAULT_AREAS + distinct.pluck(:area)).uniq.sort
  end

  def self.room_options
    where.not(room: [ nil, "" ]).distinct.order(:room).pluck(:room)
  end

  def repeating?
    repeat_unit != "none"
  end

  def complete!(user, on: Date.current)
    transaction do
      completions.create!(user: user, completed_on: on, effort: effort, planning: planning)
      repeating? ? update!(due_on: next_due_after(on)) : update!(done_at: Time.current)
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
      dates << date if date >= range.begin
      break unless repeating?

      date = advance(date)
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
      effort: effort, planning: planning, assignee_email: assignee&.email,
      done_at: done_at, last_done_on: last_done_on, updated_at: updated_at
    }
  end

  private

  def advance(date)
    case repeat_unit
    when "day" then date.advance(days: repeat_every)
    when "week" then date.advance(weeks: repeat_every)
    when "month" then date.advance(months: repeat_every)
    when "year" then date.advance(years: repeat_every)
    else date
    end
  end
end
