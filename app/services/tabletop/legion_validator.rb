module Tabletop
  # Checks a Star Wars: Legion list against its faction's imported army rules:
  # points, rank minimums and maximums, unique cards and legal upgrades.
  class LegionValidator
    attr_reader :list

    def initialize(list)
      @list = list
      @rules = list.rules
    end

    def valid?
      errors.empty?
    end

    def errors
      @errors ||= [ points_error, *faction_errors, *rank_errors, commander_error, *unique_unit_errors, *upgrade_errors ].compact
    end

    # Cards whose data is incomplete or marked to check in the import.
    def warnings
      @warnings ||= begin
        cards = entries.flat_map { |entry| [ entry.game_unit, *entry.upgrades ] }.uniq
        cards.select { |card| card.points.nil? }.map { |card| "#{card.name} has no known points yet (counted as 0)." } +
          cards.select(&:check).map { |card| "#{card.name}: card data still to verify." }
      end
    end

    def notes
      []
    end

    def tallies
      rank_requirements.map do |key, limits|
        Tally.new(rank_label(key), entries.count { |entry| rank_group(key).include?(entry.game_unit.category) }, limits["min"], limits["max"])
      end
    end

    # Upgrades that may go in this slot of this entry: right slot and unit,
    # not removed, allowed in the faction, and not already used up in the army.
    def upgrade_options(entry, slot)
      available_upgrades.fetch(slot, []).select do |upgrade|
        upgrade.fits?(entry.game_unit, alignment: @rules["alignment"]) && unique_allowed?(upgrade) &&
          entry.upgrade_ids.exclude?(upgrade.id) && (upgrade.per_army.nil? || upgrade_usage[upgrade.id].to_i < upgrade.per_army)
      end
    end

    private

    def entries
      @entries ||= list.entries.to_a
    end

    # The faction's upgrades, loaded once for every slot of the list.
    def available_upgrades
      @available_upgrades ||= GameUpgrade.available.where(game: list.game, faction: list.faction).order(:points, :name).group_by(&:slot)
    end

    # jsonb keeps no key order, so ranks are shown in the usual order.
    def rank_requirements
      (@rules["rank_requirements"] || {}).sort_by { |key, _| GameUnit.category_rank(rank_group(key).first) }
    end

    def rank_group(key)
      key.to_s.split("_or_")
    end

    def rank_label(key)
      rank_group(key).map(&:humanize).join(" + ")
    end

    def points_error
      total = list.total_points
      "#{total} points: over the #{list.points_limit} point limit." if list.points_limit && total > list.points_limit
    end

    def faction_errors
      entries.reject { |entry| entry.game_unit.faction == list.faction }.map { |entry| "#{entry.game_unit.name} is not in #{list.faction}." }
    end

    def rank_errors
      tallies.reject(&:ok?).map { |tally| "#{tally.label}: #{tally.count}, needs #{tally.range}." }
    end

    # Every Legion army is led by at least one Commander.
    def commander_error
      "Take at least one Commander." if entries.none? { |entry| entry.game_unit.category == "commander" }
    end

    def unique_unit_errors
      entries.map(&:game_unit).select(&:unique).tally.select { |_, count| count > 1 }.map { |unit, _| "#{unit.full_name} is unique: only once per army." }
    end

    def upgrade_errors
      entries.flat_map { |entry| entry_upgrade_errors(entry) } + army_upgrade_errors
    end

    def entry_upgrade_errors(entry)
      unit = entry.game_unit
      problems = entry.upgrades.flat_map do |upgrade|
        [ ("#{upgrade.name} was removed from the game." if upgrade.removed),
          ("#{upgrade.name} is a unique upgrade #{list.faction} may not take." unless unique_allowed?(upgrade)),
          ("#{upgrade.name} can't go on #{unit.name}." unless upgrade.fits?(unit, alignment: @rules["alignment"])) ]
      end
      problems += entry.upgrades.group_by(&:slot).filter_map do |slot, upgrades|
        free = unit.slots.count(slot)
        "#{unit.name} has #{free} #{slot.humanize.downcase} slot#{"s" unless free == 1}, not #{upgrades.size}." if upgrades.size > free
      end
      problems += entry.upgrades.tally.select { |_, count| count > 1 }.map { |upgrade, _| "#{unit.name} has #{upgrade.name} twice." }
      problems.compact
    end

    def army_upgrade_errors
      list.upgrades_by_id.values.filter_map do |upgrade|
        next unless upgrade.per_army && upgrade_usage[upgrade.id] > upgrade.per_army

        "#{upgrade.name} is #{upgrade.unique ? "unique" : "limited"}: at most #{upgrade.per_army} per army."
      end
    end

    def upgrade_usage
      @upgrade_usage ||= entries.flat_map(&:upgrade_ids).tally
    end

    def unique_allowed?(upgrade)
      allowed = @rules["unique_upgrades_allowed"]
      !upgrade.unique || allowed.nil? || allowed.include?(upgrade.name)
    end
  end
end
