require "test_helper"

class Household::ChoresControllerTest < ActionDispatch::IntegrationTest
  include ActiveJob::TestHelper

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
    sign_in("alice@example.com")
    assert_redirected_to household_root_path
    get household_root_path
    assert_response :success
    assert_select ".hh-chore-title", text: "Clean the toilet"
    assert_select ".hh-bar", 2
  end

  test "removing someone from the list ends their session" do
    sign_in("alice@example.com")
    ENV["HOUSEHOLD_EMAILS"] = "bob@example.com"
    get household_root_path
    assert_redirected_to household_login_path
  end

  test "filters: mine and up for grabs" do
    sign_in("bob@example.com")
    get household_root_path(show: "mine")
    assert_select ".hh-chore-title", text: "Find a GP"
    assert_select ".hh-chore-title", text: "Clean the toilet", count: 0
    get household_root_path(show: "free")
    assert_select ".hh-chore-title", text: "Find a GP", count: 0
  end

  test "add, claim, complete and delete a chore" do
    sign_in("alice@example.com")
    assert_difference -> { Chore.count } do
      post household_chores_path, params: { chore: { title: "Descale the kettle", area: "Cleaning", room: "Kitchen", due_on: Date.current, repeat_unit: "month", repeat_every: 1, effort: 1 } }
    end
    chore = Chore.find_by!(title: "Descale the kettle")
    assert_equal users(:alice), chore.created_by

    post claim_household_chore_path(chore)
    assert_equal users(:alice), chore.reload.assignee
    post claim_household_chore_path(chore)
    assert_nil chore.reload.assignee

    assert_difference -> { Completion.count } do
      post complete_household_chore_path(chore)
    end
    assert_equal Date.current.next_month, chore.reload.due_on

    delete household_chore_path(chore)
    assert_not Chore.exists?(chore.id)
  end

  test "a double tap on Done only counts once" do
    sign_in("alice@example.com")
    chore = chores(:toilet)
    seen = chore.completions.count
    assert_difference -> { Completion.count }, 1 do
      post complete_household_chore_path(chore), params: { seen: seen }
      post complete_household_chore_path(chore), params: { seen: seen }
    end
    assert_equal Date.current + 7, chore.reload.due_on
  end

  test "a one-off chore can only be done once" do
    sign_in("alice@example.com")
    chore = chores(:gp)
    assert_difference -> { Completion.count }, 1 do
      post complete_household_chore_path(chore), params: { seen: 0 }
      post complete_household_chore_path(chore), params: { seen: 0 }
      post complete_household_chore_path(chore)
    end
    assert chore.reload.done_at?
  end

  test "a double tap counts once even when done a full interval early" do
    sign_in("alice@example.com")
    chore = chores(:toilet)
    chore.update!(due_on: Date.current + 7, fixed_schedule: false)
    seen = chore.completions.count
    assert_difference -> { Completion.count }, 1 do
      post complete_household_chore_path(chore), params: { seen: seen }
      post complete_household_chore_path(chore), params: { seen: seen }
    end
  end

  test "Done can be ticked off for the person a chore belongs to" do
    sign_in("alice@example.com")
    chore = chores(:toilet)
    chore.update!(assignee: users(:bob))
    bob = users(:bob).display_name
    get household_root_path
    assert_select "button", text: "#{bob} did it"

    assert_enqueued_with(job: HouseholdPushJob, args: ->(args) { args[1].include?("ticked off by #{users(:alice).display_name}") }) do
      post complete_household_chore_path(chore), params: { seen: chore.completions.count, by: users(:bob).id }
    end
    assert_equal users(:bob), chore.completions.order(:id).last.user
    assert_equal "Clean the toilet (done by #{bob})", Event.where(action: "chore_done").last.summary
  end

  test "Done for someone else falls back to you unless they own the chore" do
    sign_in("alice@example.com")
    chore = chores(:toilet)
    post complete_household_chore_path(chore), params: { by: users(:bob).id }
    assert_equal users(:alice), chore.completions.order(:id).last.user

    chore.update!(assignee: users(:alice))
    post complete_household_chore_path(chore), params: { by: users(:bob).id }
    assert_equal users(:alice), chore.completions.order(:id).last.user
  end

  test "your own and personal chores have no did-it button" do
    chores(:toilet).update!(assignee: users(:alice))
    chores(:gp).update!(personal: true, assignee: users(:alice))
    sign_in("alice@example.com")
    get household_root_path
    assert_select "button", text: /did it\z/, count: 0
  end

  test "invalid input re-renders the form" do
    sign_in("alice@example.com")
    post household_chores_path, params: { chore: { title: "", area: "Cleaning", due_on: Date.current } }
    assert_response :unprocessable_entity
    assert_select ".hh-errors"
  end

  test "sign out" do
    sign_in("alice@example.com")
    delete household_logout_path
    get household_root_path
    assert_redirected_to household_login_path
  end

  test "a personal chore is only seen by its owner" do
    sign_in("alice@example.com")
    assert_no_enqueued_jobs(only: HouseholdPushJob) do
      post household_chores_path, params: { chore: { title: "Renew my passport", area: "Admin", due_on: Date.current, due_time: "10:00", personal: "1" } }
    end
    chore = Chore.find_by!(title: "Renew my passport")
    assert chore.personal
    assert_equal users(:alice), chore.assignee
    get household_root_path
    assert_select ".hh-chore-title", text: "Renew my passport"
    get household_agenda_path
    assert_select ".hh-chore-title", text: "Renew my passport"

    sign_in("bob@example.com")
    get household_root_path
    assert_select ".hh-chore-title", text: "Renew my passport", count: 0
    get household_agenda_path
    assert_select ".hh-chore-title", text: "Renew my passport", count: 0
    get edit_household_chore_path(chore)
    assert_response :not_found
    post complete_household_chore_path(chore)
    assert_response :not_found
    assert_nil chore.reload.done_at
  end

  test "finishing a personal chore tells no one and leaves the balance alone" do
    chore = chores(:gp)
    chore.update!(personal: true, assignee: users(:alice))
    sign_in("alice@example.com")
    assert_no_enqueued_jobs(only: HouseholdPushJob) { post complete_household_chore_path(chore) }
    assert chore.reload.done_at
    get household_root_path
    assert_select ".hh-bar-value", text: /\A0 points/, count: 2
  end

  test "a personal chore cannot be released to the household" do
    chore = chores(:gp)
    chore.update!(personal: true, assignee: users(:bob))
    sign_in("bob@example.com")
    post claim_household_chore_path(chore)
    assert_equal users(:bob), chore.reload.assignee
  end

  test "someone else's chore cannot be made personal" do
    chore = chores(:gp)
    sign_in("alice@example.com")
    patch household_chore_path(chore), params: { chore: { personal: "1" } }
    assert_response :unprocessable_entity
    assert_not chore.reload.personal
    assert_equal users(:bob), chore.assignee
  end
end
