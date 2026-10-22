# Lets Claude send a reminder to one member's phones, or everyone's.
class Household::Api::PushesController < Household::Api::BaseController
  def create
    attrs = params.expect(push: %i[to title body path])
    users = attrs[:to].blank? || attrs[:to] == "all" ? User.all : User.where(email: attrs[:to].to_s.strip.downcase)
    return render json: { errors: [ "Unknown recipient" ] }, status: :unprocessable_entity if users.none?
    return render json: { errors: [ "Title is required" ] }, status: :unprocessable_entity if attrs[:title].blank?

    path = attrs[:path].to_s.start_with?("/household") ? attrs[:path] : "/household"
    sent = PushSubscription.notify(users, title: attrs[:title], body: attrs[:body], path: path)
    render json: { devices: sent }
  end
end
