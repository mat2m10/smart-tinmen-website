require "test_helper"

class Household::ListsControllerTest < ActionDispatch::IntegrationTest
  setup do
    @old = ENV["HOUSEHOLD_EMAILS"]
    ENV["HOUSEHOLD_EMAILS"] = "alice@example.com,bob@example.com"
    OmniAuth.config.test_mode = true
  end

  teardown do
    ENV["HOUSEHOLD_EMAILS"] = @old
    OmniAuth.config.mock_auth[:google_oauth2] = nil
    OmniAuth.config.test_mode = false
  end

  def sign_in(email)
    OmniAuth.config.mock_auth[:google_oauth2] = OmniAuth::AuthHash.new(provider: "google_oauth2", uid: email, info: { email: email, first_name: "Test" })
    get "/auth/google_oauth2/callback"
  end

  test "signed-out visitors cannot see lists" do
    get household_lists_path
    assert_redirected_to household_login_path
  end

  test "the Lists tab offers the usual lists, and one tap starts one" do
    sign_in("alice@example.com")
    get household_lists_path
    assert_response :success
    assert_select ".hh-nav a[aria-current='page']", text: "Lists"
    assert_select ".hh-list-suggestions button", text: /Groceries/

    assert_difference -> { homes(:ours).lists.count }, 1 do
      post household_lists_path, params: { list: { name: "Groceries" } }
    end
    list = homes(:ours).lists.last
    assert_redirected_to household_list_path(list)

    get household_lists_path
    assert_select ".hh-list-suggestions button", text: /Groceries/, count: 0
    assert_no_difference -> { List.count } do
      post household_lists_path, params: { list: { name: " groceries " } }
    end
  end

  test "both of you add to the same list, tick things off, bring them back and clear them" do
    list = homes(:ours).lists.create!(name: "Groceries")
    sign_in("alice@example.com")
    post household_list_items_path(list), params: { list_item: { body: "Milk\n\n  Eggs  \n" } }
    assert_equal [ "Milk", "Eggs" ], list.items.map(&:body)
    assert_equal [ users(:alice) ], list.items.map(&:added_by).uniq

    sign_in("bob@example.com")
    get household_list_path(list)
    assert_select ".hh-item-body", text: "Milk"

    milk = list.items.find_by!(body: "Milk")
    patch household_list_item_path(list, milk)
    assert milk.reload.ticked?
    get household_list_path(list)
    assert_select ".hh-items--ticked .hh-item-body", text: "Milk"
    patch household_list_item_path(list, milk)
    assert_not milk.reload.ticked?

    patch household_list_item_path(list, milk)
    delete ticked_household_list_items_path(list)
    assert_equal [ "Eggs" ], list.items.reload.map(&:body)

    delete household_list_item_path(list, list.items.first)
    assert_empty list.items.reload
  end

  test "an empty item is refused" do
    list = homes(:ours).lists.create!(name: "Groceries")
    sign_in("alice@example.com")
    assert_no_difference -> { ListItem.count } do
      post household_list_items_path(list), params: { list_item: { body: "  \n " } }
    end
    assert_redirected_to household_list_path(list)
    assert flash[:alert].present?
  end

  test "deleting a list takes its items with it" do
    list = homes(:ours).lists.create!(name: "Wish list")
    list.items.create!(body: "A bike seat")
    sign_in("alice@example.com")
    assert_difference -> { ListItem.count }, -1 do
      delete household_list_path(list)
    end
    assert_redirected_to household_lists_path
  end

  test "another household's lists stay out of reach" do
    other = Home.create!(name: "Next door")
    theirs = other.lists.create!(name: "Their groceries")
    item = theirs.items.create!(body: "Their secret item")
    sign_in("alice@example.com")

    get household_lists_path
    assert_no_match "Their groceries", response.body
    get household_list_path(theirs)
    assert_response :not_found
    post household_list_items_path(theirs), params: { list_item: { body: "Sneaky" } }
    assert_response :not_found
    patch household_list_item_path(theirs, item)
    assert_response :not_found
    delete household_list_path(theirs)
    assert_response :not_found
    assert_equal [ "Their secret item" ], theirs.items.reload.map(&:body)
    assert_not item.reload.ticked?
  end
end
