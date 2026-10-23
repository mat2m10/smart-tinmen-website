# A unit card (Legion) or operative option (Kill Team). The card data comes
# from the private import; category is the Legion rank or the Kill Team role.
class GameUnit < ApplicationRecord
  # Display order of categories: Legion ranks, then Kill Team roles.
  CATEGORIES = %w[commander operative corps special_forces support heavy broodcoven extra].freeze

  has_many :army_list_entries, dependent: :restrict_with_error
  has_many :collection_items, as: :item, dependent: :destroy

  validates :game, inclusion: { in: GameFaction::GAMES.keys }
  validates :name, presence: true, uniqueness: { scope: %i[game faction] }
  validates :faction, :category, presence: true

  scope :sorted, -> { order(:points, :name) }

  def self.category_rank(category)
    CATEGORIES.index(category) || CATEGORIES.size
  end

  def full_name
    subtitle.present? ? "#{name}, #{subtitle}" : name
  end

  def unit_type
    stats["unit_type"]
  end

  def affiliation
    stats["affiliation"]
  end

  # "Transport" matches the keyword "Transport", "Jump" matches "Jump 2".
  def keyword?(word)
    keywords.any? { |keyword| keyword.casecmp?(word) || keyword.downcase.start_with?("#{word.downcase} ", "#{word.downcase}:") }
  end
end
