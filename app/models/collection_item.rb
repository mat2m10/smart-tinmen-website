# How many of a unit (or of an upgrade that adds a miniature) you own, and how
# far along they are. Owned is everything, painted or not.
class CollectionItem < ApplicationRecord
  ITEM_TYPES = %w[GameUnit GameUpgrade].freeze
  COUNTS = %i[built painted unbuilt].freeze

  belongs_to :user
  belongs_to :item, polymorphic: true

  validates :item_type, inclusion: { in: ITEM_TYPES }
  validates(*COUNTS, numericality: { only_integer: true, in: 0..999 })
  validates :note, length: { maximum: 200 }

  def owned
    built + painted + unbuilt
  end
end
