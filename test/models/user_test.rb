require "test_helper"

class UserTest < ActiveSupport::TestCase
  setup { @old = ENV["HOUSEHOLD_EMAILS"]; ENV["HOUSEHOLD_EMAILS"] = " Alice@example.com, bob@example.com " }
  teardown { ENV["HOUSEHOLD_EMAILS"] = @old }

  def google(email, first_name: "Bob")
    OmniAuth::AuthHash.new(provider: "google_oauth2", uid: "1", info: { email: email, first_name: first_name, image: "https://example.com/a.png" })
  end

  test "only allow-listed emails, case-insensitive" do
    assert User.allowed?("alice@example.com")
    assert User.allowed?("BOB@example.com")
    assert_not User.allowed?("someone@example.com")
    assert_not User.allowed?(nil)
  end

  test "nobody is allowed when the list is empty" do
    ENV["HOUSEHOLD_EMAILS"] = ""
    assert_not User.allowed?("alice@example.com")
  end

  test "from_google signs in an allowed, verified email and keeps the name fresh" do
    user = User.from_google(google("bob@example.com", first_name: "Bobby"))
    assert_equal users(:bob), user
    assert_equal "Bobby", user.reload.name
  end

  test "from_google refuses unverified (nil) and unknown emails" do
    assert_nil User.from_google(google(nil))
    assert_nil User.from_google(google("stranger@example.com"))
  end

  test "the badge counts all your open chores, also later ones, plus what is up for grabs by today" do
    alice, bob = users(:alice), users(:bob)
    # toilet: up for grabs, due today. rent: due the 25th. gp: Bob's, in ten days.
    base = Date.current >= chores(:rent).due_on ? 2 : 1
    assert_equal base, alice.badge_count
    assert_equal base + 1, bob.badge_count
    Chore.create!(title: "Bob's late chore", area: "Admin", due_on: Date.yesterday, assignee: bob)
    Chore.create!(title: "Bob's secret", area: "Admin", due_on: Date.current, assignee: bob, personal: true)
    Chore.create!(title: "Alice's late chore", area: "Admin", due_on: Date.yesterday, assignee: alice)
    Chore.create!(title: "Alice's chore next month", area: "Admin", due_on: Date.current + 30, assignee: alice)
    Chore.create!(title: "Free next week", area: "Admin", due_on: Date.current + 7)
    Chore.create!(title: "Already done", area: "Admin", due_on: Date.yesterday, done_at: Time.current)
    assert_equal base + 2, alice.badge_count
    assert_equal base + 3, bob.badge_count
  end

  test "a ticked repeating chore of yours leaves the badge until it is due again" do
    alice = users(:alice)
    chore = Chore.create!(title: "Alice's weekly", area: "Admin", due_on: Date.current, repeat_unit: "week", repeat_every: 1, assignee: alice)
    before = alice.badge_count
    chore.complete!(alice)
    assert_equal before - 1, alice.badge_count
  end
end
