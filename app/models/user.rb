# A member of the household. Only emails listed in HOUSEHOLD_EMAILS can sign in.
class User < ApplicationRecord
  has_many :assigned_chores, class_name: "Chore", foreign_key: :assignee_id, inverse_of: :assignee, dependent: :nullify
  has_many :created_chores, class_name: "Chore", foreign_key: :created_by_id, inverse_of: :created_by, dependent: :nullify
  has_many :completions, dependent: :nullify
  has_many :notes, dependent: :nullify
  has_many :push_subscriptions, dependent: :delete_all

  normalizes :email, with: ->(email) { email.strip.downcase }
  validates :email, presence: true, uniqueness: true

  def self.allowed_emails
    ENV.fetch("HOUSEHOLD_EMAILS", "").split(",").map { |email| email.strip.downcase }.reject(&:empty?)
  end

  def self.allowed?(email)
    email.present? && allowed_emails.include?(email.strip.downcase)
  end

  # Google only fills info.email when the address is verified.
  def self.from_google(auth)
    email = auth&.info&.email
    return unless allowed?(email)

    user = find_or_initialize_by(email: email)
    user.update!(name: auth.info.first_name.presence || auth.info.name.presence || user.name, avatar_url: auth.info.image)
    user
  end

  def display_name
    name.presence || email.split("@").first
  end

  def initial
    display_name.first.upcase
  end
end
