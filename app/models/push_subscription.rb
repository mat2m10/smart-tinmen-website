# One phone or browser that agreed to receive household notifications.
class PushSubscription < ApplicationRecord
  belongs_to :user

  validates :endpoint, presence: true, uniqueness: true, format: { with: %r{\Ahttps://\S+\z} }
  validates :p256dh, :auth, presence: true

  # Sends a notification to the given users' devices. Devices that the push
  # service no longer knows are removed. Returns how many were reached.
  def self.notify(users, title:, body:, path: "/household")
    keys = vapid_keys
    return 0 unless keys

    message = { title: title.to_s.truncate(80), options: { body: body.to_s.truncate(240), data: { path: path }, icon: ActionController::Base.helpers.asset_path("android-192.png") } }.to_json
    where(user: users).count do |subscription|
      WebPush.payload_send(message: message, endpoint: subscription.endpoint, p256dh: subscription.p256dh, auth: subscription.auth,
                           vapid: keys, ttl: 12.hours.to_i, urgency: "normal", open_timeout: 5, read_timeout: 10)
      true
    rescue WebPush::ExpiredSubscription, WebPush::InvalidSubscription
      subscription.destroy
      false
    rescue StandardError => e # one bad device (network, TLS, malformed keys) must not stop the others
      Rails.logger.warn("Push to subscription #{subscription.id} failed: #{e.class}")
      false
    end
  end

  def self.vapid_keys
    public_key, private_key = ENV["VAPID_PUBLIC_KEY"].presence, ENV["VAPID_PRIVATE_KEY"].presence
    { subject: "https://smarttinmen.com", public_key: public_key, private_key: private_key } if public_key && private_key
  end
end
