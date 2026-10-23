require "test_helper"

class Tabletop::KillTeamValidatorTest < ActiveSupport::TestCase
  setup { @list = army_lists(:alice_kt) }

  def add(*units, loadout: nil)
    units.each { |unit| @list.entries.create!(game_unit: game_units(unit), loadout: loadout) }
  end

  def validator
    @list.validator
  end

  def team
    add(:kt_boss, loadout: "Pistol, Knife")
    add(:kt_grunt, :kt_gunner, :kt_medic)
  end

  test "a commander with a loadout and the main operatives is valid" do
    team
    assert validator.valid?, validator.errors.inspect
    assert_equal [ "2 extra selections left." ], validator.warnings
    assert_equal [ [ "Commander", 1 ], [ "Operatives", 3 ], [ "Extra selections", 0 ], [ "Gunner + Sniper", 1 ], [ "Coven", 0 ] ],
                 validator.tallies.map { |tally| [ tally.label, tally.count ] }
  end

  test "exactly one commander, with one of its loadouts" do
    add(:kt_grunt, :kt_gunner, :kt_medic)
    assert_includes validator.errors, "Take exactly 1 commander (now 0)."
    add(:kt_boss)
    assert_includes validator.errors, "Choose a loadout for Test Boss."
    @list.entries.last.update!(loadout: "Sword, Board")
    assert_includes validator.errors, "Choose a loadout for Test Boss."
  end

  test "the main list needs its count of operatives" do
    add(:kt_boss, loadout: "Rifle, Bayonet")
    add(:kt_grunt, :kt_gunner)
    assert_equal [ "2 of 3 operatives." ], validator.errors
  end

  test "extra selections have costs, and a coven operative brings a note" do
    team
    add(:kt_witch)
    assert validator.valid?
    assert_equal [ "The commander loses LEADER." ], validator.notes
    add(:kt_pets)
    assert_includes validator.errors, "Extra selections: 3 used, only 2 allowed."
  end

  test "extra repeatable operatives and ploys use selections" do
    team
    add(:kt_grunt)
    @list.update!(ploys: 1)
    assert validator.valid?, validator.errors.inspect
    @list.update!(ploys: 2)
    assert_includes validator.errors, "Extra selections: 3 used, only 2 allowed."
    @list.update!(ploys: 3)
    assert_includes validator.errors, "At most 2 0CP ploys instead of operatives."
  end

  test "only repeatable operatives go past the main count, others only once" do
    add(:kt_boss, loadout: "Pistol, Knife")
    add(:kt_gunner, :kt_medic, :kt_medic, :kt_sniper)
    assert_includes validator.errors, "4 operatives: only Grunt operatives can go past 3."
    assert_includes validator.errors, "Test Medic can be taken only once."
  end

  test "keyword limits count every operative with one of the keywords" do
    team
    add(:kt_sniper)
    assert_includes validator.errors, "At most 1 GUNNER and SNIPER operative (now 2)."
    add(:kt_witch, :kt_seer)
    assert_includes validator.errors, "At most 1 COVEN operative (now 2)."
  end

  test "rules a faction doesn't define are not checked" do
    game_factions(:brood).update!(rules: {})
    add(:kt_grunt)
    assert validator.valid?, validator.errors.inspect
    assert_empty validator.tallies
  end
end
