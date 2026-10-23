# A list you build for one faction. Points, ranks and selections are checked
# by the game's validator against the faction's imported rules.
class ArmyList < ApplicationRecord
  belongs_to :user
  has_many :entries, -> { order(:position, :id) }, class_name: "ArmyListEntry", dependent: :delete_all, inverse_of: :army_list

  validates :game, inclusion: { in: GameFaction::GAMES.keys }
  validates :faction, presence: true
  validates :name, presence: true, length: { maximum: 80 }
  validates :points_limit, numericality: { only_integer: true, greater_than: 0 }, allow_nil: true
  validates :ploys, numericality: { only_integer: true, greater_than_or_equal_to: 0, less_than: 10 }
  validate :faction_exists, on: :create

  before_create -> { self.points_limit ||= rules["points_limit"] }

  def legion?
    game == "legion"
  end

  def game_faction
    @game_faction ||= GameFaction.find_by(game: game, name: faction)
  end

  def rules
    game_faction&.rules || {}
  end

  def validator
    (legion? ? Tabletop::LegionValidator : Tabletop::KillTeamValidator).new(self)
  end

  def total_points
    entries.sum(&:points)
  end

  # All upgrades on the list's entries, loaded once.
  def upgrades_by_id
    ids = entries.flat_map(&:upgrade_ids).uniq
    @upgrades_by_id = nil if @upgrades_by_id && (ids - @upgrades_by_id.keys).any?
    @upgrades_by_id ||= GameUpgrade.where(id: ids).index_by(&:id)
  end

  def units
    GameUnit.where(game: game, faction: faction)
  end

  private

  def faction_exists
    errors.add(:faction, "is not imported yet") unless game_faction
  end
end
