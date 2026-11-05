# One household: a couple, a family or housemates, with its own board, notes
# and balance. It is called Home because Household is the code's namespace.
# Exactly one home can have the assistant: Claude's API token, the shared Google
# Calendar and the calendar feed are set in the environment and reach only that one.
class Home < ApplicationRecord
  INVITE_DAYS = 14

  belongs_to :created_by, class_name: "User", optional: true, inverse_of: :started_homes
  has_many :users, dependent: :restrict_with_error
  has_many :chores, dependent: :restrict_with_error
  has_many :notes, dependent: :restrict_with_error
  has_many :events, dependent: :restrict_with_error

  validates :name, presence: true, length: { maximum: 60 }

  def self.assistant_home
    find_by(assistant: true)
  end

  # Where the people in HOUSEHOLD_EMAILS belong.
  def self.for_founders
    assistant_home || create!(name: "Household", assistant: true)
  end

  # The home a still-valid invite link leads to.
  def self.from_invite(token)
    where(invite_expires_at: Time.current..).find_by(invite_token: token.to_s) if token.present?
  end

  def self.fresh_invite
    { invite_token: SecureRandom.urlsafe_base64(24), invite_expires_at: INVITE_DAYS.days.from_now }
  end

  def renew_invite!
    update!(self.class.fresh_invite)
  end

  def invite_active?
    invite_token.present? && invite_expires_at&.future?
  end
end
