require "test_helper"

class Tabletop::LegionValidatorTest < ActiveSupport::TestCase
  setup { @list = army_lists(:alice_legion) }

  def add(unit, *upgrades)
    @list.entries.create!(game_unit: game_units(unit), upgrade_ids: upgrades.map { |upgrade| game_upgrades(upgrade).id })
  end

  def validator
    @list.validator
  end

  test "an empty list misses its commander and corps" do
    assert_not validator.valid?
    assert_includes validator.errors, "Commander + Operative: 0, needs 1–2."
    assert_includes validator.errors, "Corps: 0, needs 1–2."
    assert_includes validator.errors, "Take at least one Commander."
  end

  test "a commander and a corps unit within the points make a valid list" do
    add(:commander, :orders, :goggles)
    add(:troopers, :sergeant)
    assert validator.valid?, validator.errors.inspect
    assert_equal 114, @list.total_points
    assert_equal [ [ "Commander + Operative", 1, "1–2", true ], [ "Corps", 1, "1–2", true ], [ "Heavy", 0, "0–1", true ] ],
                 validator.tallies.map { |tally| [ tally.label, tally.count, tally.range, tally.ok? ] }
  end

  test "an operative alone does not lead the army, and ranks have a maximum" do
    add(:hero)
    add(:troopers)
    assert_equal [ "Take at least one Commander." ], validator.errors
    2.times { add(:commander) }
    assert_includes validator.errors, "Commander + Operative: 3, needs 1–2."
  end

  test "points over the limit are an error" do
    add(:commander)
    add(:troopers)
    add(:tank, :door_guns)
    assert_equal [ "222 points: over the 200 point limit." ], validator.errors
  end

  test "unique units and unique upgrades only once per army" do
    add(:commander)
    add(:troopers)
    add(:hero, :saber)
    add(:hero, :saber)
    assert_includes validator.errors, "Test Hero, The Brave is unique: only once per army."
    assert_includes validator.errors, "Test Saber is unique: at most 1 per army."
  end

  test "upgrades must be legal, allowed in the faction and have a free slot" do
    add(:commander, :saber, :goggles, :shiny)
    add(:troopers)
    add(:hero, :gauntlet)
    assert_includes validator.errors, "Test Saber can't go on Test Commander."
    assert_includes validator.errors, "Test Commander has 0 armament slots, not 1."
    assert_includes validator.errors, "Test Commander has 1 gear slot, not 2."
    assert_includes validator.errors, "Test Shiny Gear was removed from the game."
    assert_includes validator.errors, "Test Gauntlet is a unique upgrade Test Collective may not take."
  end

  test "an upgrade twice on the same unit is refused" do
    add(:commander, :orders, :orders)
    add(:troopers)
    assert_includes validator.errors, "Test Commander has Test Orders twice."
  end

  test "upgrades without points are counted as 0 and flagged" do
    add(:commander, :mystery)
    assert_includes validator.warnings, "Test Mystery Gear has no known points yet (counted as 0)."
    assert_includes validator.warnings, "Test Mystery Gear: card data still to verify."
  end

  test "slot options only offer legal, available and unused upgrades" do
    commander = add(:commander)
    assert_equal [ "Test Goggles", "Test Mystery Gear" ], validator.upgrade_options(commander, "gear").map(&:name)
    assert_equal [ "Test Sergeant" ], validator.upgrade_options(add(:troopers), "personnel").map(&:name)
    assert_equal [ "Test Driver", "Test Door Guns" ], validator.upgrade_options(add(:tank), "pilot").map(&:name)

    hero = add(:hero)
    assert_equal [ "Test Fury", "Test Lead", "Test Stance" ], validator.upgrade_options(hero, "training").map(&:name)
    assert_equal [ "Test Saber" ], validator.upgrade_options(hero, "armament").map(&:name)
    hero.update!(upgrade_ids: [ game_upgrades(:saber).id ])
    assert_empty @list.validator.upgrade_options(add(:hero), "armament")
  end

  test "upgrades fill the slots of their kind in order" do
    entry = add(:troopers, :goggles, :sergeant, :orders)
    assert_equal [ [ "personnel", "Test Sergeant" ], [ "gear", "Test Goggles" ] ], entry.filled_slots.map { |slot, upgrade| [ slot, upgrade&.name ] }
    assert_equal [ "Test Orders" ], entry.unslotted_upgrades.map(&:name)
  end
end
