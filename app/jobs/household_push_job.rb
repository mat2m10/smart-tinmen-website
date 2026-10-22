# Sends household notifications outside the request.
class HouseholdPushJob < ApplicationJob
  def perform(user_ids, title, body, path)
    PushSubscription.notify(User.where(id: user_ids), title: title, body: body, path: path)
  end
end
