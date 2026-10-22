# A private iCalendar feed of open chores, to subscribe to in Google Calendar.
# The URL carries a secret key (HOUSEHOLD_CALENDAR_TOKEN, at least 32
# characters) because calendar apps can't sign in.
# ?only=planning limits it to planning tasks.
class Household::CalendarsController < ActionController::Base
  def show
    expected = ENV["HOUSEHOLD_CALENDAR_TOKEN"].to_s
    return head :not_found unless expected.length >= 32 && ActiveSupport::SecurityUtils.secure_compare(params[:key].to_s, expected)

    chores = Chore.pending.includes(:assignee).order(:due_on, :title)
    chores = chores.where(planning: true) if params[:only] == "planning"
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    response.headers["Cache-Control"] = "private, max-age=900"
    send_data Household::Ical.new(chores, name: params[:only] == "planning" ? "Household · planning" : "Household").to_s,
              type: "text/calendar; charset=utf-8", disposition: "inline", filename: "household.ics"
  end
end
