require "test_helper"

class Household::ApiListsTest < ActionDispatch::IntegrationTest
  TOKEN = "t" * 40

  setup { @old = ENV["HOUSEHOLD_API_TOKEN"]; ENV["HOUSEHOLD_API_TOKEN"] = TOKEN }
  teardown { ENV["HOUSEHOLD_API_TOKEN"] = @old }

  def auth
    { "Authorization" => "Bearer #{TOKEN}" }
  end

  test "needs the token" do
    get household_api_lists_path
    assert_response :unauthorized
  end

  test "Claude reads the lists of the assistant home only" do
    groceries = homes(:ours).lists.create!(name: "Groceries")
    groceries.items.create!(body: "Milk", added_by: users(:alice))
    groceries.items.create!(body: "Eggs", checked_at: Time.current)
    Home.create!(name: "Next door").lists.create!(name: "Their list")

    get household_api_lists_path, headers: auth
    assert_response :success
    assert_equal [ "Groceries" ], response.parsed_body.map { |list| list["name"] }
    items = response.parsed_body.first["items"]
    assert_equal [ [ "Milk", false, "alice@example.com" ], [ "Eggs", true, nil ] ], items.map { |item| item.values_at("body", "ticked", "added_by") }
  end

  test "Claude adds items to an existing list, matched without case" do
    groceries = homes(:ours).lists.create!(name: "Groceries")
    post household_api_lists_path, headers: auth, as: :json, params: { list: { name: "groceries", items: [ " Onions ", "Garlic", "" ] } }
    assert_response :created
    assert_equal [ "Onions", "Garlic" ], groceries.items.reload.map(&:body)
    assert_nil groceries.items.first.added_by
  end

  test "Claude starts a list that does not exist yet" do
    assert_difference -> { homes(:ours).lists.count }, 1 do
      post household_api_lists_path, headers: auth, as: :json, params: { list: { name: "Gift ideas", items: [ "Book" ] } }
    end
    assert_response :created
    assert_equal "Gift ideas", response.parsed_body["name"]
  end

  test "nothing to add is refused" do
    assert_no_difference -> { List.count } do
      post household_api_lists_path, headers: auth, as: :json, params: { list: { name: "Groceries", items: [ " " ] } }
    end
    assert_response :unprocessable_entity
  end
end
