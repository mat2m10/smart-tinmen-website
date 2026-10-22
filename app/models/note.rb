# Something one of us (or Claude) wrote down: a remark on a chore, or a loose
# note for the household with optional photos and a voice note.
class Note < ApplicationRecord
  MAX_PHOTOS = 4

  belongs_to :chore, optional: true, inverse_of: :remarks
  belongs_to :user, optional: true
  has_many :attachments, -> { without_data.order(:id) }, dependent: :delete_all, inverse_of: :note

  validates :body, length: { maximum: 5000 }
  validate :has_content

  scope :recent_first, -> { order(created_at: :desc, id: :desc) }
  # Loose notes, and remarks on chores the user may see (not the other's personal ones).
  scope :visible_to, ->(user) { where(chore_id: nil).or(where(chore_id: Chore.visible_to(user).select(:id))) }

  def author_name
    from_claude ? "Claude" : (user&.display_name || "Someone")
  end

  def author_initial
    from_claude ? "✦" : (user&.initial || "?")
  end

  def photos = attachments.select(&:photo?)
  def voices = attachments.select(&:voice?)

  def as_api_json
    {
      id: id, chore_id: chore_id, chore_title: chore&.title, body: body,
      author: from_claude ? "claude" : user&.email, created_at: created_at,
      attachments: attachments.map(&:as_api_json)
    }
  end

  private

  def has_content
    errors.add(:base, "Write something, or add a photo or a voice note") if body.blank? && attachments.empty?
  end
end
