require "test_helper"

class Tabletop::ImporterTest < ActiveSupport::TestCase
  LEGION = <<~YAML
    meta:
      game: "Star Wars: Legion (made up)"
      faction: Fake Syndicate (Battle Force)
    army_rules:
      points_limit: 600
      alignment: Dark Side
      rank_requirements_standard:
        commander_or_operative: {min: 1, max: 2}
        corps: {min: 1, max: 3}
      unique_upgrades_allowed: [Fake Blade]
    units:
      - {name: Fake Boss, title: The Loud, rank: commander, unit_type: Trooper, affiliation: Fakers, points: 60, points_printed: 65,
         minis: 1, unique: true, upgrade_bar: [command, gear], keywords: ["Jump 1"], weapons: [Fist], note: "Shouts a lot"}
      - {name: Fake Grunts, rank: corps, points: 40, minis: 4, upgrade_bar: [personnel], check: true}
    upgrades:
      - {name: Fake Blade, slot: armament, points: 5, restriction: Fake Boss, unique: true, adds_mini: false}
      - {name: Fake Old Card, slot: gear, points: 3, restriction: null}
      - {name: Fake Uplink, slot: comms, points: 10, limited: 2}
    removed_cards_relevant: [Fake Old Card]
  YAML

  KILLTEAM = <<~YAML
    meta:
      game: "Kill Team (made up)"
      kill_team: Fake Cult
    selection:
      leader:
        default: Fake Leader (LEADER)
        broodcoven_option: >
          One COVEN operative may be taken.
          The Leader then loses LEADER.
      commander:
        count: 1
        loadout_options:
          - [Pistol, Knife]
          - [Rifle]
      main_list:
        count: 4
        options: [Fake Gunner with Flamer and Bayonet, Fake Gunner with Melta and Bayonet, Fake Grunt]
      extra_selections:
        count: 2
        options:
          - {name: Fake Pet, counts_as: 1, note: "you get 2 Fake Pets for one selection"}
          - {name: Fake Witch, counts_as: 2, broodcoven: true}
          - {name: Fake Grunt, counts_as: 1}
          - {name: "0CP ploy", counts_as: 1, note: "max 2 times"}
      limits:
        - "Other than GRUNT, each option can be included only once."
        - "Max 1 GUNNER operative."
        - "Bring snacks."
    operatives:
      - {name: Fake Leader, keywords: [CULT, LEADER], apl: 2, move: 6, save: "5+", wounds: 8, weapons: [{name: Pistol}, {name: Knife}]}
      - {name: Fake Gunner, keywords: [CULT, GUNNER], weapons: [{name: Flamer}, {name: "Melta (short)"}, {name: Bayonet}]}
      - {name: Fake Grunt, keywords: [CULT, GRUNT], weapons: [{name: Gun}]}
      - {name: Fake Pet, keywords: [CULT, PET]}
      - {name: Fake Witch, keywords: [CULT, COVEN], weapons: [{name: Staff, check: true}]}
  YAML

  test "imports a Legion faction with its rules, units and upgrades, and updates on a second run" do
    result = Tabletop::Importer.new(LEGION).run
    assert_equal "Star Wars: Legion · Fake Syndicate: rules 1 new, 0 updated; units 2 new, 0 updated; upgrades 3 new, 0 updated.", result.to_s

    rules = GameFaction.find_by!(game: "legion", name: "Fake Syndicate").rules
    assert_equal 600, rules["points_limit"]
    assert_equal({ "min" => 1, "max" => 3 }, rules["rank_requirements"]["corps"])
    assert_equal [ "Fake Blade" ], rules["unique_upgrades_allowed"]

    boss = GameUnit.find_by!(game: "legion", faction: "Fake Syndicate", name: "Fake Boss")
    assert_equal [ "The Loud", "commander", 60, 65, true ], [ boss.subtitle, boss.category, boss.points, boss.points_printed, boss.unique ]
    assert_equal [ "command", "gear" ], boss.slots
    assert_equal "Fakers", boss.affiliation
    assert_equal "Shouts a lot", boss.notes
    assert GameUnit.find_by!(name: "Fake Grunts").check
    assert GameUpgrade.find_by!(name: "Fake Old Card").removed
    assert_equal 2, GameUpgrade.find_by!(name: "Fake Uplink").per_army

    again = Tabletop::Importer.new(LEGION.sub("points: 60", "points: 55")).run
    assert_equal [ 0, 2 ], again.counts["units"]
    assert_equal 55, boss.reload.points
    assert_equal 2, GameUnit.where(faction: "Fake Syndicate").count
  end

  test "imports a Kill Team with its selection rules and one unit per option" do
    Tabletop::Importer.new(KILLTEAM).run
    rules = GameFaction.find_by!(game: "killteam", name: "Fake Cult").rules
    assert_equal [ "Pistol, Knife", "Rifle" ], rules["loadouts"]
    assert_equal [ 1, 4, 2, 1, 2 ], rules.values_at("commander_count", "main_count", "extra_count", "ploy_cost", "max_ploys")
    assert_equal "GRUNT", rules["repeatable_keyword"]
    assert_equal [ { "keywords" => [ "GUNNER" ], "max" => 1 } ], rules["keyword_limits"]
    assert_equal [ "Bring snacks." ], rules["other_limits"]
    assert_equal "One COVEN operative may be taken. The Leader then loses LEADER.", rules["broodcoven_note"]

    units = GameUnit.where(game: "killteam", faction: "Fake Cult").index_by(&:name)
    assert_equal [ "Fake Gunner with Flamer and Bayonet", "Fake Gunner with Melta and Bayonet", "Fake Grunt", "Fake Leader", "Fake Pet", "Fake Witch" ].sort, units.keys.sort
    assert_equal "commander", units["Fake Leader"].category
    assert_equal [ "Flamer", "Bayonet" ], units["Fake Gunner with Flamer and Bayonet"].stats["weapons"].pluck("name")
    assert_equal [ "Melta (short)", "Bayonet" ], units["Fake Gunner with Melta and Bayonet"].stats["weapons"].pluck("name")
    assert_equal [ "operative", 1 ], [ units["Fake Grunt"].category, units["Fake Grunt"].selection_cost ]
    assert_equal [ "extra", 1, 2 ], [ units["Fake Pet"].category, units["Fake Pet"].selection_cost, units["Fake Pet"].minis ]
    assert_equal [ "broodcoven", 2, true ], [ units["Fake Witch"].category, units["Fake Witch"].selection_cost, units["Fake Witch"].check ]
    assert_nil units["Fake Leader"].points
  end

  test "a file for an unknown game is refused" do
    assert_raises(ArgumentError) { Tabletop::Importer.new("meta: {game: Chess}").run }
  end
end
