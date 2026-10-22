# Stores (or forgets) the push subscription of the phone the member is using.
class Household::PushSubscriptionsController < Household::BaseController
  def create
    attrs = params.expect(subscription: [ :endpoint, { keys: %i[p256dh auth] } ])
    subscription = PushSubscription.find_or_initialize_by(endpoint: attrs[:endpoint])
    subscription.assign_attributes(user: current_user, p256dh: attrs.dig(:keys, :p256dh), auth: attrs.dig(:keys, :auth), user_agent: request.user_agent.to_s.truncate(250))
    if subscription.save
      head :created
    else
      render json: { errors: subscription.errors.full_messages }, status: :unprocessable_entity
    end
  end

  def destroy
    current_user.push_subscriptions.where(endpoint: params[:endpoint]).delete_all
    head :no_content
  end
end
