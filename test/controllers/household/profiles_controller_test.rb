require "test_helper"

class Household::ProfilesControllerTest < ActionDispatch::IntegrationTest
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

  test "you can pick your name and colour, and the name survives the next sign-in" do
    sign_in("alice@example.com")
    get household_profile_path
    assert_select "input[name='user[color]']", minimum: 8

    patch household_profile_path, params: { user: { nickname: " Ali ", color: "#3B6EA8" } }
    assert_redirected_to household_profile_path
    assert_equal "Ali", users(:alice).reload.display_name
    assert_equal "#3B6EA8", users(:alice).color

    sign_in("alice@example.com")
    assert_equal "Ali", users(:alice).reload.display_name
    get household_root_path
    assert_select ".hh-me-link .hh-avatar", text: "A"
  end

  test "a bad colour is refused" do
    sign_in("alice@example.com")
    patch household_profile_path, params: { user: { color: "red; background: url(x)" } }
    assert_response :unprocessable_entity
    assert_nil users(:alice).reload.color
  end

  test "an empty name falls back to the Google name" do
    sign_in("alice@example.com")
    patch household_profile_path, params: { user: { nickname: "  " } }
    assert_equal "Test", users(:alice).reload.display_name
  end

  test "a profile photo shows as the avatar, can be served to the household and removed" do
    sign_in("alice@example.com")
    patch household_profile_path, params: { user: { photo_upload: fixture_file_upload("photo.jpg", "image/jpeg") } }
    alice = users(:alice).reload
    assert alice.photo?
    assert_equal "image/jpeg", alice.photo_type

    get household_root_path
    assert_select ".hh-me-link .hh-avatar--photo img[src^='#{household_avatar_path(alice)}']"
    sign_in("bob@example.com")
    get household_avatar_path(alice, v: alice.updated_at.to_i)
    assert_response :success
    assert_equal "image/jpeg", response.media_type

    sign_in("alice@example.com")
    patch household_profile_path, params: { user: { remove_photo: "1" } }
    assert_not users(:alice).reload.photo?
    get household_avatar_path(alice)
    assert_response :not_found
  end

  test "a file that is not a picture is refused as a photo" do
    sign_in("alice@example.com")
    patch household_profile_path, params: { user: { photo_upload: fixture_file_upload("not_a_photo.jpg", "image/jpeg") } }
    assert_response :unprocessable_entity
    assert_not users(:alice).reload.photo?
  end

  test "avatars are only served to signed-in members" do
    users(:alice).update!(photo: file_fixture("photo.jpg").binread, photo_type: "image/jpeg")
    get household_avatar_path(users(:alice))
    assert_redirected_to household_login_path
  end

  test "signed out, the profile is not reachable" do
    get household_profile_path
    assert_redirected_to household_login_path
  end
end
