module Tabletop
  # Checks a Kill Team roster against its faction's imported selection rules:
  # the commander and its loadout, the main operatives, the extra selections
  # and their costs, options taken once and keyword limits (like at most 2
  # GUNNER + SCOUT). A rule the faction doesn't define is not checked.
  class KillTeamValidator
    attr_reader :list

    def initialize(list)
      @list = list
      @rules = list.rules
    end

    def valid?
      errors.empty?
    end

    def errors
      @errors ||= [ *faction_errors, commander_error, *loadout_errors, main_error, extra_error, ploy_error, *repeat_errors, *keyword_errors ].compact
    end

    def warnings
      @warnings ||= begin
        left = extra_count.to_i - extra_used
        [ ("#{left} extra selection#{"s" unless left == 1} left." if extra_count && left.positive?),
          *entries.map(&:game_unit).uniq.select(&:check).map { |unit| "#{unit.name}: card data still to verify." } ].compact
      end
    end

    def notes
      [ (@rules["broodcoven_note"] if broodcoven?), *@rules["other_limits"] ].compact
    end

    def tallies
      [ (Tally.new("Commander", commanders.size, commander_count, commander_count) if commander_count),
        (Tally.new("Operatives", [ operatives.size, main_count ].min, main_count, main_count) if main_count),
        (Tally.new("Extra selections", extra_used, 0, extra_count) if extra_count),
        *keyword_limits.map { |limit| Tally.new(limit["keywords"].join(" + ").titleize, keyword_count(limit["keywords"]), 0, limit["max"]) } ].compact
    end

    def loadouts
      @rules["loadouts"] || []
    end

    def max_ploys
      @rules["max_ploys"] || extra_count.to_i
    end

    private

    def entries
      @entries ||= list.entries.to_a
    end

    def in_category(category)
      entries.select { |entry| entry.game_unit.category == category }
    end

    def commanders
      in_category("commander")
    end

    def operatives
      in_category("operative")
    end

    def commander_count
      @rules["commander_count"]
    end

    def main_count
      @rules["main_count"]
    end

    def extra_count
      @rules["extra_count"]
    end

    def keyword_limits
      @rules["keyword_limits"] || []
    end

    def repeatable?(unit)
      @rules["repeatable_keyword"].present? && unit.keyword?(@rules["repeatable_keyword"])
    end

    # Operatives past the main count can only be repeatable ones bought as extra selections.
    def extra_operatives
      main_count ? [ operatives.size - main_count, 0 ].max : 0
    end

    def extra_used
      repeat_cost = operatives.map(&:game_unit).find { |unit| repeatable?(unit) }&.selection_cost || 1
      extra_operatives * repeat_cost +
        entries.reject { |entry| %w[commander operative].include?(entry.game_unit.category) }.sum { |entry| entry.game_unit.selection_cost || 1 } +
        list.ploys * @rules["ploy_cost"].to_i
    end

    def broodcoven?
      entries.any? { |entry| entry.game_unit.category == "broodcoven" }
    end

    def faction_errors
      entries.reject { |entry| entry.game_unit.faction == list.faction }.map { |entry| "#{entry.game_unit.name} is not in #{list.faction}." }
    end

    def commander_error
      "Take exactly #{commander_count} commander (now #{commanders.size})." if commander_count && commanders.size != commander_count
    end

    def loadout_errors
      commanders.reject { |entry| loadouts.empty? || loadouts.include?(entry.loadout) }.map { |entry| "Choose a loadout for #{entry.game_unit.name}." }
    end

    def main_error
      return unless main_count

      if operatives.size < main_count
        "#{operatives.size} of #{main_count} operatives."
      elsif extra_operatives > operatives.count { |entry| repeatable?(entry.game_unit) }
        "#{operatives.size} operatives: only #{@rules["repeatable_keyword"].to_s.titleize} operatives can go past #{main_count}."
      end
    end

    def extra_error
      "Extra selections: #{extra_used} used, only #{extra_count} allowed." if extra_count && extra_used > extra_count
    end

    def ploy_error
      "At most #{max_ploys} 0CP ploys instead of operatives." if list.ploys > max_ploys
    end

    def repeat_errors
      entries.map(&:game_unit).reject { |unit| repeatable?(unit) }.tally.select { |_, count| count > 1 }.map { |unit, _| "#{unit.name} can be taken only once." }
    end

    def keyword_errors
      keyword_limits.filter_map do |limit|
        count = keyword_count(limit["keywords"])
        "At most #{limit["max"]} #{limit["keywords"].join(" and ")} #{"operative".pluralize(limit["max"])} (now #{count})." if count > limit["max"]
      end
    end

    def keyword_count(keywords)
      entries.count { |entry| keywords.any? { |keyword| entry.game_unit.keyword?(keyword) } }
    end
  end
end
