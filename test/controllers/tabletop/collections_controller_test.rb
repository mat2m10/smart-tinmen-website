require "test_helper"

class Tabletop::CollectionsControllerTest < ActionDispatch::IntegrationTest
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

  test "signed out, the collection is not reachable" do
    get tabletop_collection_path
    assert_redirected_to household_login_path
    patch tabletop_collection_path, params: { item_type: "GameUnit", item_id: game_units(:tank).id, collection_item: { built: 1 } }
    assert_redirected_to household_login_path
  end

  test "shows units and upgrade miniatures per game with steppers" do
    sign_in("alice@example.com")
    get tabletop_collection_path(game: "legion")
    assert_response :success
    assert_select ".tt-stock .hh-chore-title", text: "Test Hero, The Brave"
    assert_select ".tt-stock .hh-chore-title", text: "Test Sergeant"
    assert_select ".tt-stock .hh-chore-title", text: "Test Goggles", count: 0
    assert_select "#item-game-unit-#{game_units(:troopers).id} input[name='collection_item[painted]'][value='1']"
    assert_select "form[data-controller=stepper][data-action='submit->stepper#submit'] button[data-stepper-delta-param='1']", minimum: 3
    assert_select "[data-stepper-target=owned]", text: "2"

    get tabletop_collection_path(game: "killteam")
    assert_select ".tt-stock .hh-chore-title", text: "Test Witch"
  end

  test "the steppers save in the background and get the new counts as JSON" do
    sign_in("alice@example.com")
    patch tabletop_collection_path, params: { item_type: "GameUnit", item_id: game_units(:troopers).id, collection_item: { built: 2, painted: 3, unbuilt: 0 } },
                                    headers: { "Accept" => "application/json" }
    assert_response :success
    assert_equal "application/json", response.media_type
    assert_equal({ "built" => 2, "painted" => 3, "unbuilt" => 0, "owned" => 5 }, response.parsed_body)
    assert_equal 5, collection_items(:alice_troopers).reload.owned

    patch tabletop_collection_path, params: { item_type: "GameUnit", item_id: game_units(:troopers).id, collection_item: { painted: 1000 } },
                                    headers: { "Accept" => "application/json" }
    assert_response :unprocessable_entity
    assert_match(/Painted/, response.parsed_body["errors"].join)
  end

  test "counts are saved per member" do
    sign_in("alice@example.com")
    patch tabletop_collection_path, params: { item_type: "GameUpgrade", item_id: game_upgrades(:sergeant).id, collection_item: { built: 0, painted: 2, unbuilt: 1 } }
    assert_redirected_to tabletop_collection_path(game: "legion")
    item = users(:alice).collection_items.find_by!(item: game_upgrades(:sergeant))
    assert_equal [ 0, 2, 1, 3 ], [ item.built, item.painted, item.unbuilt, item.owned ]

    patch tabletop_collection_path, params: { item_type: "GameUnit", item_id: game_units(:troopers).id, collection_item: { built: -1 } }
    assert_match(/Built/, flash[:alert])
    assert_equal 1, collection_items(:alice_troopers).reload.built

    sign_in("bob@example.com")
    get tabletop_collection_path(game: "legion")
    assert_select "#item-game-unit-#{game_units(:troopers).id} input[name='collection_item[painted]'][value='0']"
  end
end
