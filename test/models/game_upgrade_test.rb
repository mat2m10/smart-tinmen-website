require "test_helper"

class GameUpgradeTest < ActiveSupport::TestCase
  def fits?(upgrade, unit)
    game_upgrades(upgrade).fits?(game_units(unit), alignment: "Dark Side")
  end

  test "an upgrade without restriction fits any unit" do
    assert fits?(:goggles, :commander)
    assert fits?(:goggles, :tank)
  end

  test "a restriction can name the unit, its rank, its type or its affiliation and type" do
    assert fits?(:sergeant, :troopers)
    assert_not fits?(:sergeant, :commander)
    assert fits?(:lead, :hero)
    assert fits?(:lead, :commander)
    assert_not fits?(:lead, :troopers)
    assert fits?(:vehicle_comms, :tank)
    assert_not fits?(:vehicle_comms, :troopers)
    assert fits?(:driver, :tank)
    assert_not fits?(:driver, :troopers)
  end

  test "a restriction can ask for a keyword, an upgrade icon or the army's alignment" do
    assert fits?(:door_guns, :tank)
    assert_not fits?(:door_guns, :troopers)
    assert fits?(:stance, :hero)
    assert_not fits?(:stance, :commander)
    assert fits?(:fury, :hero)
    assert_not game_upgrades(:fury).fits?(game_units(:hero), alignment: "Light Side")
  end

  test "a restriction the builder doesn't understand never fits" do
    assert_not fits?(:weird, :troopers)
    assert_not fits?(:weird, :commander)
  end

  test "unique means once per army, limited as printed" do
    assert_equal 1, game_upgrades(:saber).per_army
    assert_equal 2, game_upgrades(:uplink).per_army
    assert_nil game_upgrades(:goggles).per_army
  end
end
