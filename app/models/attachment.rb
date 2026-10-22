# A photo or voice note stored in the database. Only image and audio types are
# accepted, checked against the file's own bytes, so nothing else can be served back.
class Attachment < ApplicationRecord
  TYPES = {
    "photo" => %w[image/jpeg image/png image/webp image/heic image/heif],
    "voice" => %w[audio/webm audio/ogg audio/mp4 audio/x-m4a audio/mpeg audio/aac audio/wav audio/x-wav video/webm video/mp4]
  }.freeze
  MAX_BYTES = { "photo" => 8.megabytes, "voice" => 10.megabytes }.freeze

  belongs_to :note

  validates :kind, inclusion: { in: TYPES.keys }
  validates :content_type, presence: true
  validates :byte_size, numericality: { greater_than: 0 }
  validate :type_and_size_allowed

  # Leaves the file bytes out. Use .size or .length on these, not .count:
  # COUNT over a column list is not valid SQL.
  scope :without_data, -> { select(column_names - [ "data" ]) }

  # Builds an attachment from an uploaded file. The type comes from the bytes,
  # not from what the browser claims.
  def self.from_upload(kind, file)
    return new(kind: kind, content_type: file.content_type.to_s, byte_size: file.size) if file.size > MAX_BYTES.fetch(kind, 0)

    data = file.read
    detected = Marcel::MimeType.for(StringIO.new(data), name: file.original_filename, declared_type: file.content_type)
    # Chrome records audio as WebM, which looks like video/webm to a sniffer.
    detected = detected.sub("video/", "audio/") if kind == "voice"
    new(kind: kind, content_type: detected, byte_size: data.bytesize, data: data)
  end

  def photo? = kind == "photo"
  def voice? = kind == "voice"

  def as_api_json
    { id: id, kind: kind, content_type: content_type, byte_size: byte_size, transcript: transcript }
  end

  private

  def type_and_size_allowed
    return unless TYPES.key?(kind)

    allowed = TYPES[kind].map { |type| type.sub("video/", "audio/") }
    errors.add(:base, "That file type can't be added as a #{kind}") unless allowed.include?(content_type)
    errors.add(:base, "That #{kind} is too large") if byte_size.to_i > MAX_BYTES[kind]
  end
end
