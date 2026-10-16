require "test_helper"

class UserTest < ActiveSupport::TestCase
  setup { @old = ENV["HOUSEHOLD_EMAILS"]; ENV["HOUSEHOLD_EMAILS"] = " Matthieu@example.com, livia@example.com " }
  teardown { ENV["HOUSEHOLD_EMAILS"] = @old }

  def google(email, first_name: "Livia")
    OmniAuth::AuthHash.new(provider: "google_oauth2", uid: "1", info: { email: email, first_name: first_name, image: "https://example.com/a.png" })
  end

  test "only allow-listed emails, case-insensitive" do
    assert User.allowed?("matthieu@example.com")
    assert User.allowed?("LIVIA@example.com")
    assert_not User.allowed?("someone@example.com")
    assert_not User.allowed?(nil)
  end

  test "nobody is allowed when the list is empty" do
    ENV["HOUSEHOLD_EMAILS"] = ""
    assert_not User.allowed?("matthieu@example.com")
  end

  test "from_google signs in an allowed, verified email and keeps the name fresh" do
    user = User.from_google(google("livia@example.com", first_name: "Liv"))
    assert_equal users(:livia), user
    assert_equal "Liv", user.reload.name
  end

  test "from_google refuses unverified (nil) and unknown emails" do
    assert_nil User.from_google(google(nil))
    assert_nil User.from_google(google("stranger@example.com"))
  end
end
