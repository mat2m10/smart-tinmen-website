# Adding and removing units on a list, and filling their upgrade slots or loadout.
class Tabletop::EntriesController < Tabletop::BaseController
  before_action -> { @list = find_list(params[:list_id]) }

  def create
    unit = @list.units.find(params[:game_unit_id])
    @list.entries.create!(game_unit: unit)
    redirect_to back_path, notice: "#{unit.name} added", status: :see_other
  end

  def update
    entry = @list.entries.find(params[:id])
    if params.key?(:loadout)
      entry.update!(loadout: params[:loadout].presence)
    else
      return redirect_to back_path, alert: "That upgrade doesn't fit there.", status: :see_other unless swap_upgrade(entry)
    end
    redirect_to back_path, status: :see_other
  end

  def destroy
    entry = @list.entries.find(params[:id])
    entry.destroy!
    redirect_to back_path, notice: "#{entry.game_unit.name} removed", status: :see_other
  end

  private

  # Puts an upgrade in a slot, replacing the one there (or empties the slot).
  # Only upgrades the builder offers for that slot are accepted.
  def swap_upgrade(entry)
    slot = params[:slot].to_s
    old_id = entry.upgrade_ids.find { |id| id.to_s == params[:replace_id].to_s }
    upgrade = GameUpgrade.find_by(id: params[:upgrade_id]) if params[:upgrade_id].present?
    return false unless entry.game_unit.slots.include?(slot)
    return false if upgrade && @list.validator.upgrade_options(entry, slot).exclude?(upgrade)

    entry.swap_upgrade(old_id, upgrade&.id)
    entry.save!
  end

  def back_path
    tabletop_list_path(@list, owned: params[:owned].presence)
  end
end
