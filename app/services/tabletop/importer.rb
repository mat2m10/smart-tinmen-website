module Tabletop
  # Loads one faction's cards and building rules from the private YAML files
  # (Legion: army_rules, units, upgrades; Kill Team: selection, operatives).
  # Safe to run again: records are updated by game + faction + name, never deleted.
  class Importer
    Result = Struct.new(:game, :faction, :counts) do
      def to_s
        "#{GameFaction::GAMES[game]} · #{faction}: " + counts.map { |kind, (created, updated)| "#{kind} #{created} new, #{updated} updated" }.join("; ") + "."
      end
    end

    def initialize(yaml)
      @data = YAML.safe_load(yaml, permitted_classes: [ Date ]) || {}
      @game = detect_game
      @counts = Hash.new { |hash, key| hash[key] = [ 0, 0 ] }
    end

    def run
      ActiveRecord::Base.transaction { @game == "legion" ? import_legion : import_killteam }
      Result.new(@game, @faction, @counts)
    end

    private

    def detect_game
      name = @data["game"] || @data.dig("meta", "game").to_s
      return "killteam" if name.match?(/kill\s*team/i)
      return "legion" if name.match?(/legion/i)

      raise ArgumentError, "Can't tell the game: set meta.game to a Legion or Kill Team name"
    end

    # "Some Faction (Battle Force)" becomes "Some Faction".
    def without_brackets(name)
      name.to_s.sub(/\s*\(.*\)\z/, "")
    end

    def save(model, kind, name, attrs)
      keys = { game: @game, name: name }
      keys[:faction] = @faction unless model == GameFaction
      record = model.find_or_initialize_by(keys)
      record.update!(attrs)
      @counts[kind][record.previously_new_record? ? 0 : 1] += 1
    end

    def save_faction(name, rules)
      @faction = without_brackets(name).presence || raise(ArgumentError, "meta.faction (Legion) or meta.kill_team is missing")
      save(GameFaction, "rules", @faction, rules: rules)
    end

    def import_legion
      rules = @data.fetch("army_rules")
      save_faction(@data.dig("meta", "faction"),
                   "points_limit" => rules["points_limit"], "alignment" => rules["alignment"],
                   "rank_requirements" => rules["rank_requirements_standard"], "unique_upgrades_allowed" => rules["unique_upgrades_allowed"])
      Array(@data["units"]).each { |row| save(GameUnit, "units", row.fetch("name"), legion_unit(row)) }
      removed = Array(@data["removed_cards_relevant"])
      Array(@data["upgrades"]).each { |row| save(GameUpgrade, "upgrades", row.fetch("name"), legion_upgrade(row, removed)) }
    end

    def legion_unit(row)
      { subtitle: row["title"], category: row.fetch("rank"), points: row["points"], points_printed: row["points_printed"],
        minis: row["minis"] || 1, unique: row["unique"] == true, slots: Array(row["upgrade_bar"]), keywords: Array(row["keywords"]),
        stats: row.slice("unit_type", "affiliation", "wounds", "courage", "resilience", "defense", "weapons"),
        notes: row["note"], check: row["check"] == true }
    end

    def legion_upgrade(row, removed)
      { slot: row.fetch("slot"), points: row["points"], points_printed: row["points_printed"], restriction: row["restriction"],
        unique: row["unique"] == true, limited: row["limited"], adds_mini: row["adds_mini"] == true, minis_added: row["minis_added"],
        removed: removed.include?(row["name"]), notes: row["note"], check: row["check"] == true }
    end

    # A section of the selection rules, or {} when it is missing or plain
    # text (a team without extra selections says "extra_selections: none").
    def section(selection, key)
      value = selection[key]
      value.is_a?(Hash) ? value : {}
    end

    # The leader's count and loadouts sit under a key named after it
    # ("commander:", "prime:"): the first other section with loadout options.
    def leader_section(selection)
      selection.except("leader", "main_list", "extra_selections").values.find { |value| value.is_a?(Hash) && value.key?("loadout_options") } || {}
    end

    def import_killteam
      selection = @data.fetch("selection")
      @cards = Array(@data["operatives"]).index_by { |card| card.fetch("name") }
      extras = Array(section(selection, "extra_selections")["options"]).select { |option| option.is_a?(Hash) }
      ploy = extras.find { |option| option["name"].to_s.match?(/ploy/i) }
      save_faction(@data.dig("meta", "kill_team"), killteam_rules(selection, ploy))

      costs = extras.to_h { |option| [ option["name"], option["counts_as"] ] }
      leader = without_brackets(section(selection, "leader")["default"]).presence
      save_operative(leader, "commander") if leader
      main = Array(section(selection, "main_list")["options"])
      main.each { |name| save_operative(name, "operative", selection_cost: costs[name]) }
      (extras - [ ploy ]).reject { |option| main.include?(option["name"]) }.each do |option|
        save_operative(option["name"], option["broodcoven"] ? "broodcoven" : "extra", selection_cost: option["counts_as"],
                       minis: option["note"].to_s[/get (\d+)/, 1]&.to_i || 1, note: option["note"])
      end
    end

    def killteam_rules(selection, ploy)
      leader = leader_section(selection)
      { "commander_count" => leader["count"] || 1,
        "loadouts" => Array(leader["loadout_options"]).map { |loadout| Array(loadout).join(", ") },
        "main_count" => section(selection, "main_list")["count"], "extra_count" => section(selection, "extra_selections")["count"],
        "ploy_cost" => ploy&.dig("counts_as"), "max_ploys" => ploy&.dig("note").to_s[/max (\d+) times/i, 1]&.to_i,
        "broodcoven_note" => section(selection, "leader")["broodcoven_option"]&.squish }.merge(killteam_limits(selection["limits"]))
    end

    # "Other than GRUNT, each option (or operative) can be included only once." and
    # "Max 2 GUNNER and SCOUT operatives combined." become structured rules;
    # any other line is kept as text and shown under the list.
    def killteam_limits(lines)
      rules = { "repeatable_keyword" => nil, "keyword_limits" => [], "other_limits" => [] }
      Array(lines).each do |line|
        if (match = line.to_s.match(/\AOther than (\S+), each (?:option|operative) can be included only once/i))
          rules["repeatable_keyword"] = match[1]
        elsif (match = line.to_s.match(/\AMax (\d+) (.+?) operatives?\b/i))
          rules["keyword_limits"] << { "keywords" => match[2].split(/\s+and\s+|,\s*/), "max" => match[1].to_i }
        else
          rules["other_limits"] << line
        end
      end
      rules
    end

    # Options like "Cult Gunner with Flamer and Knife" use the "Cult Gunner"
    # datacard with only the weapons they name.
    def save_operative(name, category, selection_cost: nil, minis: 1, note: nil)
      card = @cards[name] || @cards.values.select { |c| name.start_with?("#{c["name"]} ") }.max_by { |c| c["name"].size }
      raise ArgumentError, "No operative datacard for #{name}" unless card

      weapons = Array(card["weapons"])
      gear = name.split(" with ", 2)[1]
      weapons = weapons.select { |weapon| gear.include?(without_brackets(weapon["name"])) } if gear
      save(GameUnit, "operatives", name,
           category: category, minis: minis, selection_cost: selection_cost, keywords: Array(card["keywords"]),
           stats: card.slice("apl", "move", "save", "wounds", "base_mm", "abilities", "unique_actions").merge("weapons" => weapons),
           notes: [ card["note"], note ].compact.join(" ").presence, check: card["check"] == true || weapons.any? { |weapon| weapon["check"] })
    end
  end
end
