# A Legion upgrade card. Removed cards stay in the data but are never offered.
class GameUpgrade < ApplicationRecord
  has_many :collection_items, as: :item, dependent: :destroy

  validates :game, inclusion: { in: GameFaction::GAMES.keys }
  validates :name, presence: true, uniqueness: { scope: %i[game faction] }
  validates :faction, :slot, presence: true

  scope :available, -> { where(removed: false) }

  # How many copies one army may hold: unique (•) once, limited (••) as printed.
  def per_army
    unique ? 1 : limited
  end

  # Restrictions are short card texts: a unit name ("Some Squad"), name and title
  # ("Some Hero, The Brave"), affiliation and type ("Pirates Vehicle"), ranks
  # ("Commander or Operative"), "unit with Transport", "Dark Side". A unit fits when one " or " part names it, its rank, its type
  # (with or without its affiliation in front), a keyword or one of its slots.
  # Anything else does not fit, so an unknown text never lets a card through.
  def fits?(unit, alignment: nil)
    return true if restriction.blank?

    restriction.sub(/\s*\(.*\)\z/, "").split(/\s+or\s+/i).any? { |term| term_fits?(term.strip, unit, alignment) }
  end

  private

  def term_fits?(term, unit, alignment)
    return true if alignment.present? && term.casecmp?(alignment)

    if (wanted = term[/\Aunit with (?:\d\+ )?(.+?)(?: icons?)?\z/i, 1])
      unit.keyword?(wanted) || unit.slots.include?(wanted.parameterize(separator: "_"))
    else
      [ unit.name, unit.full_name, unit.category.humanize ].any? { |name| name.casecmp?(term) } || type_fits?(term, unit)
    end
  end

  def type_fits?(term, unit)
    type = unit.unit_type.to_s.downcase
    return false if type.empty?

    [ term, (term.delete_prefix("#{unit.affiliation} ") if unit.affiliation.present?) ].compact.uniq.any? do |wanted|
      type == wanted.downcase || type.end_with?(" #{wanted.downcase}")
    end
  end
end
