require "test_helper"

class Household::CompletionsControllerTest < ActionDispatch::IntegrationTest
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

  test "the Done page lists recent ticks and undo puts a repeating chore back" do
    chore = chores(:toilet)
    due = chore.due_on
    chore.complete!(users(:alice))
    sign_in("alice@example.com")
    get household_completions_path
    assert_select ".hh-chore-title", text: "Clean the toilet"

    assert_difference -> { Completion.count }, -1 do
      delete household_completion_path(chore.completions.last)
    end
    assert_equal due, chore.reload.due_on
    assert_equal "chore_undone", Event.last.action
  end

  test "undoing a tick from before previous_due_on existed puts the chore on the day it was done" do
    chore = chores(:toilet)
    chore.complete!(users(:alice), on: Date.current - 2)
    chore.completions.last.update!(previous_due_on: nil)
    sign_in("alice@example.com")
    delete household_completion_path(chore.completions.last)
    assert_equal Date.current - 2, chore.reload.due_on
  end

  test "a personal tick cannot be moved to the other member" do
    chore = chores(:gp)
    chore.update!(personal: true, assignee: users(:alice))
    chore.complete!(users(:alice))
    sign_in("alice@example.com")
    patch household_completion_path(chore.completions.last), params: { completion: { user_id: users(:bob).id } }
    assert_response :not_found
    assert_equal users(:alice), chore.completions.last.reload.user
  end

  test "undo reopens a one-off chore" do
    chore = chores(:gp)
    chore.complete!(users(:bob))
    assert chore.reload.done_at
    sign_in("alice@example.com")
    delete household_completion_path(chore.completions.last)
    assert_nil chore.reload.done_at
  end

  test "undoing an older tick leaves the chore's date alone" do
    chore = chores(:toilet)
    chore.complete!(users(:alice))
    older = chore.completions.last
    chore.complete!(users(:alice))
    due = chore.reload.due_on
    sign_in("alice@example.com")
    delete household_completion_path(older)
    assert_equal due, chore.reload.due_on
  end

  test "a tick can be put on the other member" do
    chore = chores(:toilet)
    chore.complete!(users(:alice))
    sign_in("alice@example.com")
    patch household_completion_path(chore.completions.last), params: { completion: { user_id: users(:bob).id } }
    assert_equal users(:bob), chore.completions.last.reload.user
  end

  test "ticks older than three weeks and someone else's personal ticks are not listed" do
    old = chores(:toilet)
    old.complete!(users(:alice), on: Date.current - 30)
    secret = chores(:gp)
    secret.update!(personal: true, assignee: users(:bob))
    secret.complete!(users(:bob))
    sign_in("alice@example.com")
    get household_completions_path
    assert_select ".hh-chore-title", count: 0
    delete household_completion_path(secret.completions.last)
    assert_response :not_found
  end
end
