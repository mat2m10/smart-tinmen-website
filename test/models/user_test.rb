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
end
