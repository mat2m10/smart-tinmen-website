# Sends household notifications outside the request. Production uses the
# in-process :async adapter, so a notification queued during a dyno restart is
# lost; that is acceptable for best-effort notifications.
class HouseholdPushJob < ApplicationJob
  def perform(user_ids, title, body, path)
    PushSubscription.notify(User.where(id: user_ids), title: title, body: body, path: path)
  end
end
