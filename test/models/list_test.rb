require "test_helper"

class ListTest < ActiveSupport::TestCase
  test "a home cannot have two lists with the same name" do
    homes(:ours).lists.create!(name: "Groceries")
    assert_not homes(:ours).lists.new(name: "  GROCERIES ").valid?
    assert Home.create!(name: "Next door").lists.new(name: "Groceries").valid?
  end

  test "items keep the order they were added in and know when they are ticked" do
    list = homes(:ours).lists.create!(name: "Groceries")
    first = list.items.create!(body: " Milk ")
    list.items.create!(body: "Eggs", checked_at: Time.current)
    assert_equal "Milk", first.body
    assert_equal [ "Milk", "Eggs" ], list.items.reload.map(&:body)
    assert_equal [ "Eggs" ], list.items.ticked.map(&:body)
    assert_equal [ "Milk" ], list.items.unticked.map(&:body)
  end
end
