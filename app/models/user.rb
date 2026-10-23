# A member of the household. Only emails listed in HOUSEHOLD_EMAILS can sign in.
class User < ApplicationRecord
  has_many :assigned_chores, class_name: "Chore", foreign_key: :assignee_id, inverse_of: :assignee, dependent: :nullify
  has_many :created_chores, class_name: "Chore", foreign_key: :created_by_id, inverse_of: :created_by, dependent: :nullify
  has_many :completions, dependent: :nullify
  has_many :notes, dependent: :nullify
  has_many :push_subscriptions, dependent: :delete_all

  normalizes :email, with: ->(email) { email.strip.downcase }
  validates :email, presence: true, uniqueness: true
  validates :color, format: { with: /\A#\h{6}\z/ }, allow_nil: true
  validates :nickname, length: { maximum: 30 }

  # Colours offered on the profile page; any other hex set earlier still works.
  COLORS = { "#3F8F5A" => "Green", "#B7472A" => "Rust", "#3B6EA8" => "Blue", "#2F8A8A" => "Teal",
             "#8A5BA8" => "Purple", "#B8487A" => "Rose", "#C28A1E" => "Ochre", "#5E4B3C" => "Brown" }.freeze

  normalizes :nickname, with: ->(value) { value.strip.presence }

  PHOTO_TYPES = %w[image/jpeg image/png image/webp].freeze
  PHOTO_MAX_BYTES = 2.megabytes
  validate :photo_allowed

  # Takes an uploaded picture; the type comes from the bytes, not the browser.
  def photo_upload=(file)
    return unless file.respond_to?(:read)

    data = file.read
    self.photo_type = Marcel::MimeType.for(StringIO.new(data))
    self.photo = data
  end

  def remove_photo=(value)
    self.photo = self.photo_type = nil if ActiveModel::Type::Boolean.new.cast(value)
  end

  def photo?
    photo_type.present?
  end

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
    nickname.presence || name.presence || email.split("@").first
  end

  def initial
    display_name.first.upcase
  end

  private

  def photo_allowed
    return if photo.nil?

    errors.add(:photo, "must be a JPEG, PNG or WebP picture") unless PHOTO_TYPES.include?(photo_type)
    errors.add(:photo, "is too large") if photo.bytesize > PHOTO_MAX_BYTES
  end
end
