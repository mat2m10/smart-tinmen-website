require "test_helper"

class Household::ChoresControllerTest < ActionDispatch::IntegrationTest
  setup do
    @old = ENV["HOUSEHOLD_EMAILS"]
    ENV["HOUSEHOLD_EMAILS"] = "matthieu@example.com,livia@example.com"
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

  test "signed-out visitors are sent to the login page, which stays out of search" do
    get household_root_path
    assert_redirected_to household_login_path
    get household_login_path
    assert_response :success
    assert_equal "noindex, nofollow", response.headers["X-Robots-Tag"]
    assert_select "meta[name='robots'][content='noindex, nofollow']"
  end

  test "a Google account that is not on the list cannot get in" do
    sign_in("stranger@example.com")
    assert_redirected_to household_login_path
    get household_root_path
    assert_redirected_to household_login_path
  end

  test "a household member sees the board" do
    sign_in("matthieu@example.com")
    assert_redirected_to household_root_path
    get household_root_path
    assert_response :success
    assert_select ".hh-chore-title", text: "Clean the toilet"
    assert_select ".hh-bar", 2
  end

  test "removing someone from the list ends their session" do
    sign_in("matthieu@example.com")
    ENV["HOUSEHOLD_EMAILS"] = "livia@example.com"
    get household_root_path
    assert_redirected_to household_login_path
  end

  test "filters: mine and up for grabs" do
    sign_in("livia@example.com")
    get household_root_path(show: "mine")
    assert_select ".hh-chore-title", text: "Find a GP"
    assert_select ".hh-chore-title", text: "Clean the toilet", count: 0
    get household_root_path(show: "free")
    assert_select ".hh-chore-title", text: "Find a GP", count: 0
  end

  test "add, claim, complete and delete a chore" do
    sign_in("matthieu@example.com")
    assert_difference -> { Chore.count } do
      post household_chores_path, params: { chore: { title: "Descale the kettle", area: "cleaning", room: "Kitchen", due_on: Date.current, repeat_unit: "month", repeat_every: 1, effort: 1 } }
    end
    chore = Chore.find_by!(title: "Descale the kettle")
    assert_equal users(:matthieu), chore.created_by

    post claim_household_chore_path(chore)
    assert_equal users(:matthieu), chore.reload.assignee
    post claim_household_chore_path(chore)
    assert_nil chore.reload.assignee

    assert_difference -> { Completion.count } do
      post complete_household_chore_path(chore)
    end
    assert_equal Date.current.next_month, chore.reload.due_on

    delete household_chore_path(chore)
    assert_not Chore.exists?(chore.id)
  end

  test "invalid input re-renders the form" do
    sign_in("matthieu@example.com")
    post household_chores_path, params: { chore: { title: "", area: "cleaning", due_on: Date.current } }
    assert_response :unprocessable_entity
    assert_select ".hh-errors"
  end

  test "sign out" do
    sign_in("matthieu@example.com")
    delete household_logout_path
    get household_root_path
    assert_redirected_to household_login_path
  end
end
