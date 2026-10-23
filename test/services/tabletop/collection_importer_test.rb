require "test_helper"

class Tabletop::CollectionImporterTest < ActiveSupport::TestCase
  test "sets counts for units and upgrade miniatures and reports unknown names" do
    yaml = <<~YAML
      game: legion
      items:
        - {name: "Test Troopers", painted: 4, built: 1}
        - {name: "Test Sergeant", painted: 1, note: "leader for the troopers"}
        - {name: "Test Tank", built: 1, unbuilt: 1}
        - {name: "Not A Real Unit", built: 2}
    YAML
    result = Tabletop::CollectionImporter.new(users(:alice), yaml).run

    assert_equal 3, result.updated
    assert_equal [ "Not A Real Unit" ], result.unknown
    collection = Tabletop::Collection.new(users(:alice), "legion")
    assert_equal [ 5, 4 ], [ collection.owned(game_units(:troopers)), collection.painted(game_units(:troopers)) ]
    assert_equal 0, collection.item(game_units(:troopers)).unbuilt
    assert_equal "leader for the troopers", collection.item(game_upgrades(:sergeant)).note
    assert_equal 2, collection.owned(game_units(:tank))
    assert_equal 0, Tabletop::Collection.new(users(:bob), "legion").owned(game_units(:tank))
  end

  test "names are looked up in the given game only" do
    result = Tabletop::CollectionImporter.new(users(:alice), "game: killteam\nitems:\n  - {name: Test Troopers, built: 1}\n").run
    assert_equal [ "Test Troopers" ], result.unknown
  end

  test "the game must be named" do
    assert_raises(ArgumentError) { Tabletop::CollectionImporter.new(users(:alice), "items: []") }
  end
end
