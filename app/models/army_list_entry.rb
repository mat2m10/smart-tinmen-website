# One unit or operative in a list, with the Legion upgrades it carries
# (ids into game_upgrades) or the Kill Team loadout it was given.
class ArmyListEntry < ApplicationRecord
  belongs_to :army_list, touch: true
  belongs_to :game_unit

  validates :loadout, length: { maximum: 120 }

  before_create -> { self.position = (ArmyListEntry.where(army_list_id: army_list_id).maximum(:position) || 0) + 1 }

  def upgrades
    upgrade_ids.filter_map { |id| army_list.upgrades_by_id[id] }
  end

  def points
    game_unit.points.to_i + upgrades.sum { |upgrade| upgrade.points.to_i }
  end

  # The unit's upgrade bar in order, each slot holding the next upgrade of its kind.
  def filled_slots
    left = upgrades.group_by(&:slot)
    game_unit.slots.map { |slot| [ slot, left[slot]&.shift ] }
  end

  # Upgrades that found no slot of their kind on the bar.
  def unslotted_upgrades
    upgrades - filled_slots.filter_map(&:last)
  end

  def swap_upgrade(old_id, new_id)
    ids = upgrade_ids.dup
    index = old_id && ids.index(old_id)
    ids.delete_at(index) if index
    ids << new_id if new_id
    self.upgrade_ids = ids
  end
end
