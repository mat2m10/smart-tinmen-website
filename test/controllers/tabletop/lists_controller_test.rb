require "test_helper"

class Tabletop::ListsControllerTest < ActionDispatch::IntegrationTest
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

  test "signed out, the list builder sends you to sign in and back" do
    get tabletop_root_path
    assert_redirected_to household_login_path
    get print_tabletop_list_path(army_lists(:alice_legion))
    assert_redirected_to household_login_path
    post tabletop_lists_path, params: { army_list: { name: "x" } }
    assert_redirected_to household_login_path

    sign_in("alice@example.com")
    assert_redirected_to print_tabletop_list_path(army_lists(:alice_legion))
  end

  test "a member only sees and changes their own lists" do
    sign_in("alice@example.com")
    bobs = army_lists(:bob_legion)
    get tabletop_list_path(bobs)
    assert_response :not_found
    get print_tabletop_list_path(bobs)
    assert_response :not_found
    post tabletop_list_entries_path(bobs), params: { game_unit_id: game_units(:commander).id }
    assert_response :not_found
    delete tabletop_list_path(bobs)
    assert_response :not_found
    assert ArmyList.exists?(bobs.id)

    get tabletop_root_path
    assert_select ".hh-chore-title", text: "Alice's strike force"
    assert_select ".hh-chore-title", text: "Bob's army", count: 0
  end

  test "builds a Legion list: add units, fill slots, see totals, remove a unit" do
    sign_in("alice@example.com")
    get new_tabletop_list_path
    assert_select "input[name='army_list[faction_id]']", 2
    post tabletop_lists_path, params: { army_list: { name: "Friday", faction_id: game_factions(:collective).id } }
    list = users(:alice).army_lists.find_by!(name: "Friday")
    assert_redirected_to tabletop_list_path(list)
    assert_equal [ "legion", "Test Collective", 200 ], [ list.game, list.faction, list.points_limit ]

    get tabletop_list_path(list)
    assert_select ".tt-badge--bad"
    assert_select ".tt-pick .tt-pick-name", text: /Test Troopers/

    post tabletop_list_entries_path(list), params: { game_unit_id: game_units(:commander).id }
    post tabletop_list_entries_path(list), params: { game_unit_id: game_units(:troopers).id }
    assert_redirected_to tabletop_list_path(list)
    commander = list.entries.find_by!(game_unit: game_units(:commander))

    patch tabletop_list_entry_path(list, commander), params: { slot: "gear", upgrade_id: game_upgrades(:goggles).id }
    assert_equal [ game_upgrades(:goggles).id ], commander.reload.upgrade_ids
    patch tabletop_list_entry_path(list, commander), params: { slot: "gear", replace_id: game_upgrades(:goggles).id, upgrade_id: game_upgrades(:shiny).id }
    assert_equal "That upgrade doesn't fit there.", flash[:alert]
    patch tabletop_list_entry_path(list, commander), params: { slot: "armament", upgrade_id: game_upgrades(:saber).id }
    assert_equal [ game_upgrades(:goggles).id ], commander.reload.upgrade_ids

    get tabletop_list_path(list)
    assert_select ".tt-badge--ok"
    assert_select ".tt-total", text: "94 / 200 pts"
    assert_select ".tt-slot--filled .tt-slot-name", text: /Test Goggles/
    assert_select ".tt-tallies li", text: /Corps\s+1/

    patch tabletop_list_entry_path(list, commander), params: { slot: "gear", replace_id: game_upgrades(:goggles).id, upgrade_id: "" }
    assert_empty commander.reload.upgrade_ids

    delete tabletop_list_entry_path(list, commander)
    assert_equal [ game_units(:troopers) ], list.entries.reload.map(&:game_unit)
    get tabletop_root_path
    assert_select ".tt-list", text: /Friday.*2 problems/m
  end

  test "units from another faction cannot be added" do
    sign_in("alice@example.com")
    post tabletop_list_entries_path(army_lists(:alice_legion)), params: { game_unit_id: game_units(:kt_grunt).id }
    assert_response :not_found
  end

  test "a Kill Team list takes a commander loadout and ploys" do
    sign_in("alice@example.com")
    list = army_lists(:alice_kt)
    boss = list.entries.create!(game_unit: game_units(:kt_boss))
    get tabletop_list_path(list)
    assert_select "button[name=loadout][value='Pistol, Knife']"
    patch tabletop_list_entry_path(list, boss), params: { loadout: "Pistol, Knife" }
    assert_equal "Pistol, Knife", boss.reload.loadout
    patch tabletop_list_path(list), params: { army_list: { ploys: 1 } }
    assert_equal 1, list.reload.ploys
    get tabletop_list_path(list)
    assert_select ".tt-ploys strong", text: "1"
    assert_select ".tt-problems li", text: "0 of 3 operatives."
  end

  test "the list shows what you own, and can show only owned units" do
    sign_in("alice@example.com")
    list = army_lists(:alice_legion)
    list.entries.create!(game_unit: game_units(:troopers))
    list.entries.create!(game_unit: game_units(:troopers))
    list.entries.create!(game_unit: game_units(:tank))
    get tabletop_list_path(list)
    assert_select ".tt-entry .tt-own", text: "Own 2 · 1 painted"
    assert_select ".tt-entry .tt-own--short", text: "Not owned"

    get tabletop_list_path(list, owned: "1")
    assert_select ".tt-pick .tt-pick-name", count: 1
    assert_select ".tt-pick .tt-pick-name", text: /Test Troopers/
  end

  test "the printable card lists units, upgrades and points" do
    sign_in("alice@example.com")
    list = army_lists(:alice_legion)
    list.entries.create!(game_unit: game_units(:troopers), upgrade_ids: [ game_upgrades(:sergeant).id ])
    get print_tabletop_list_path(list)
    assert_response :success
    assert_select ".tt-print h1", text: "Alice's strike force"
    assert_select ".tt-print header", text: /55 \/ 200 pts/
    assert_select ".tt-print-sub", text: "Test Sergeant (15)"
    assert_select "[data-controller=print]"
  end

  test "lists can be renamed and deleted" do
    sign_in("alice@example.com")
    list = army_lists(:alice_legion)
    get edit_tabletop_list_path(list)
    assert_response :success
    patch tabletop_list_path(list), params: { army_list: { name: "Renamed", points_limit: 800 } }
    assert_equal [ "Renamed", 800 ], list.reload.values_at(:name, :points_limit)
    patch tabletop_list_path(list), params: { army_list: { name: "" } }
    assert_response :unprocessable_entity
    delete tabletop_list_path(list)
    assert_redirected_to tabletop_root_path
    assert_not ArmyList.exists?(list.id)
  end
end
