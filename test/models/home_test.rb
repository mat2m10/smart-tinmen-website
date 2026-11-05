require "test_helper"

class HomeTest < ActiveSupport::TestCase
  setup { @old = ENV["HOUSEHOLD_EMAILS"]; ENV["HOUSEHOLD_EMAILS"] = "new-founder@example.com" }
  teardown { ENV["HOUSEHOLD_EMAILS"] = @old }

  def google(email)
    OmniAuth::AuthHash.new(provider: "google_oauth2", uid: "1", info: { email: email, first_name: "Test" })
  end

  test "a founder who signs in for the first time lands in the home with the assistant" do
    assert_equal homes(:ours), User.from_google(google("new-founder@example.com")).home
  end

  test "an invite only counts while it is valid" do
    home = Home.create!(name: "Friends", **Home.fresh_invite)
    assert_equal home, Home.from_invite(home.invite_token)
    home.update!(invite_expires_at: 1.second.ago)
    assert_nil Home.from_invite(home.invite_token)
    assert_nil Home.from_invite("")
    assert_nil Home.from_invite(nil)
  end

  test "a chore can't be given to someone in another home" do
    other = Home.create!(name: "Next door")
    carol = User.create!(email: "carol@example.com", home: other)
    chore = homes(:ours).chores.new(title: "x", area: "Admin", due_on: Date.current, assignee: carol)
    assert_not chore.valid?
    assert chore.errors[:assignee].any?
  end

  test "only one home can have the assistant" do
    assert_raises(ActiveRecord::RecordNotUnique) { Home.create!(name: "Second", assistant: true) }
  end
end
