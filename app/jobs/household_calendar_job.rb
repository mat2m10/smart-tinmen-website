# Mirrors one chore into the shared Google Calendar: an event while it has a
# time, none once the time is removed, the chore is deleted or it is made
# personal. A done one-off chore keeps its event (it records the appointment);
# a repeating one moves its event to the next date. Like HouseholdPushJob it runs on
# the in-process :async adapter, so a sync lost in a restart waits for the
# chore's next change.
class HouseholdCalendarJob < ApplicationJob
  def perform(chore_id)
    calendar = Household::GoogleCalendar.from_env
    return unless calendar

    chore = Chore.includes(:assignee).find_by(id: chore_id)
    chore&.due_time && !chore.personal ? calendar.upsert(chore) : calendar.delete(chore_id)
  end
end
