# Two weeks of chores, day by day. Chores that repeat weekly or less often
# also appear on their coming dates, so you can see what next week looks like;
# daily and weekday ones only on their real due date, or they would fill every day.
class Household::AgendaController < Household::BaseController
  DAYS = 14

  def show
    today = Date.current
    @from = parse_date(params[:from])&.clamp(today - 365, today + 365) || today
    @days = @from...(@from + DAYS)
    chores = Chore.visible_to(current_user).pending.includes(:assignee).order(:due_on, :title).to_a
    @overdue = @from == today ? chores.select { |chore| chore.due_on < today } : []

    @entries = Hash.new { |hash, key| hash[key] = [] }
    chores.each do |chore|
      dates = chore.repeat_unit.in?(%w[day weekday]) ? [ chore.due_on ].select { |date| @days.cover?(date) } : chore.occurrences(@days)
      dates.each { |date| @entries[date] << { chore: chore, date: date, forecast: date != chore.due_on } }
    end
    @entries.each_value { |list| list.sort_by! { |entry| [ entry[:chore].due_time || "", entry[:chore].title ] } }
  end

  private

  def parse_date(value)
    Date.iso8601(value) if value.present?
  rescue Date::Error
    nil
  end
end
