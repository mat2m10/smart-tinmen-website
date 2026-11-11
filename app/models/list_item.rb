# One thing on a list. Ticking it keeps it greyed out at the bottom until
# someone clears the ticked items, so a tick by mistake is easy to undo.
class ListItem < ApplicationRecord
  belongs_to :list, touch: true
  belongs_to :added_by, class_name: "User", optional: true

  validates :body, presence: true, length: { maximum: 200 }

  normalizes :body, with: ->(body) { body.squish }

  scope :unticked, -> { where(checked_at: nil) }
  scope :ticked, -> { where.not(checked_at: nil) }

  def ticked?
    checked_at.present?
  end
end
