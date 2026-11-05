require "test_helper"

# Two households on one site: nothing of one may reach the other.
class Household::HomesTest < ActionDispatch::IntegrationTest
  include ActiveJob::TestHelper

  TOKEN = "t" * 40

  setup do
    @old_emails, @old_token, @old_calendar = ENV.values_at("HOUSEHOLD_EMAILS", "HOUSEHOLD_API_TOKEN", "HOUSEHOLD_CALENDAR_TOKEN")
    ENV["HOUSEHOLD_EMAILS"] = "alice@example.com"
    ENV["HOUSEHOLD_API_TOKEN"] = TOKEN
    ENV["HOUSEHOLD_CALENDAR_TOKEN"] = TOKEN
    OmniAuth.config.test_mode = true

    @other = Home.create!(name: "Next door")
    @carol = User.create!(email: "carol@example.com", name: "Carol", home: @other)
    @secret = @other.chores.create!(title: "Next door's chore", area: "Next door area", room: "Next door room", due_on: Date.current, assignee: @carol)
    @their_note = @other.notes.create!(user: @carol, body: "Next door's note")
    Event.log!("chore_added", user: @carol, chore: @secret)
  end

  teardown do
    ENV["HOUSEHOLD_EMAILS"], ENV["HOUSEHOLD_API_TOKEN"], ENV["HOUSEHOLD_CALENDAR_TOKEN"] = @old_emails, @old_token, @old_calendar
    OmniAuth.config.mock_auth[:google_oauth2] = nil
    OmniAuth.config.test_mode = false
  end

  def sign_in(email)
    OmniAuth.config.mock_auth[:google_oauth2] = OmniAuth::AuthHash.new(provider: "google_oauth2", uid: email, info: { email: email, first_name: "Test" })
    get "/auth/google_oauth2/callback"
  end

  def api
    { "Authorization" => "Bearer #{TOKEN}" }
  end

  test "the board, agenda, done page and notes show only your own home" do
    @secret.complete!(@carol)
    sign_in("alice@example.com")
    [ household_root_path, household_agenda_path, household_completions_path, household_notes_path ].each do |path|
      get path
      assert_response :success
      assert_no_match "Next door", response.body, path
      assert_no_match "Carol", response.body, path
    end
    get new_household_chore_path
    assert_no_match "Next door", response.body
    assert_no_match "Carol", response.body
  end

  test "the balance counts only your own home" do
    sign_in("carol@example.com")
    get household_root_path
    assert_select ".hh-bar", 1
    assert_select ".hh-chore-title", text: "Clean the toilet", count: 0
    assert_select ".hh-chore-title", text: "Next door's chore"
  end

  test "another home's chore can't be opened, ticked, claimed, assigned or deleted" do
    sign_in("alice@example.com")
    get edit_household_chore_path(@secret)
    assert_response :not_found
    post complete_household_chore_path(@secret)
    assert_response :not_found
    post claim_household_chore_path(@secret)
    assert_response :not_found
    post assign_household_chore_path(@secret), params: { assignee_id: users(:bob).id }
    assert_response :not_found
    delete household_chore_path(@secret)
    assert_response :not_found
    assert_equal @carol, @secret.reload.assignee
    assert_empty @secret.completions
  end

  test "you can't hand work to someone in another home" do
    sign_in("alice@example.com")
    chore = chores(:toilet)
    post assign_household_chore_path(chore), params: { assignee_id: @carol.id }
    assert_response :not_found
    assert_no_difference -> { Chore.count } do
      post household_chores_path, params: { chore: { title: "Sneaky", area: "Admin", due_on: Date.current, assignee_id: @carol.id } }
    end
    assert_response :unprocessable_entity
    patch household_chore_path(chore), params: { chore: { assignee_id: @carol.id } }
    assert_response :unprocessable_entity
    assert_nil chore.reload.assignee
  end

  test "another home's notes, ticks and pictures can't be reached" do
    attachment = @their_note.attachments.create!(kind: "voice", content_type: "audio/webm", byte_size: 3, data: "abc")
    tick = @secret.complete!(@carol)
    @carol.update!(photo: "abc", photo_type: "image/png")
    sign_in("alice@example.com")
    get household_attachment_path(attachment)
    assert_response :not_found
    get household_avatar_path(@carol)
    assert_response :not_found
    delete household_note_path(@their_note)
    assert_response :not_found
    patch household_completion_path(tick), params: { completion: { user_id: users(:alice).id } }
    assert_response :not_found
    delete household_completion_path(tick)
    assert_response :not_found
    post household_notes_path, params: { note: { body: "Hi", chore_id: @secret.id } }
    assert_response :not_found
    assert Note.exists?(@their_note.id)
    assert Completion.exists?(tick.id)
  end

  test "a note and a tick only notify your own home" do
    sign_in("carol@example.com")
    assert_no_enqueued_jobs(only: HouseholdPushJob) { post household_notes_path, params: { note: { body: "Hello" } } }
    assert_equal @other, Note.last.home
    assert_equal @other, Event.last.home
  end

  test "the API token reaches only the home with the assistant" do
    get household_api_chores_path(include_done: "true"), headers: api
    assert_no_match "Next door", response.body
    get household_api_notes_path, headers: api
    assert_no_match "Next door", response.body
    get household_api_events_path, headers: api
    assert_no_match "Next door", response.body
    @secret.complete!(@carol)
    get household_api_completions_path, headers: api
    assert_no_match "Next door", response.body

    patch household_api_chore_path(@secret), params: { chore: { title: "Changed" } }, headers: api
    assert_response :not_found
    post complete_household_api_chore_path(@secret), params: { completion: { by: "carol@example.com" } }, headers: api
    assert_response :not_found
    post household_api_chores_path, params: { chore: { title: "x", area: "Admin", due_on: Date.current, assignee_email: "carol@example.com" } }, headers: api
    assert_response :unprocessable_entity
    post household_api_notes_path, params: { note: { body: "x", chore_id: @secret.id } }, headers: api
    assert_response :unprocessable_entity
    post household_api_push_path, params: { push: { to: "carol@example.com", title: "x" } }, headers: api
    assert_response :unprocessable_entity

    attachment = @their_note.attachments.create!(kind: "voice", content_type: "audio/webm", byte_size: 3, data: "abc")
    get household_api_attachment_path(attachment), headers: api
    assert_response :not_found
    patch household_api_attachment_path(attachment), params: { attachment: { transcript: "x" } }, headers: api
    assert_response :not_found
  end

  test "the calendar feed and calendar sync leave other homes out" do
    get household_calendar_path(key: TOKEN)
    assert_response :success
    assert_includes response.body, "Clean the toilet"
    assert_no_match "Next door", response.body

    assert_no_enqueued_jobs(only: HouseholdCalendarJob) do
      @other.chores.create!(title: "Their appointment", area: "Admin", due_on: Date.current, due_time: "10:00")
    end
    assert_enqueued_jobs(1, only: HouseholdCalendarJob) do
      homes(:ours).chores.create!(title: "Our appointment", area: "Admin", due_on: Date.current, due_time: "10:00")
    end
  end

  test "an invite link brings a new person into that home" do
    @other.renew_invite!
    get household_join_path(@other.invite_token)
    assert_response :success
    assert_select "h1", text: "Join Next door"
    sign_in("dave@example.com")
    assert_redirected_to household_root_path
    assert_equal @other, User.find_by!(email: "dave@example.com").home
    get household_root_path
    assert_select ".hh-chore-title", text: "Next door's chore"
    assert_select ".hh-chore-title", text: "Clean the toilet", count: 0
  end

  test "several people can join on the same link" do
    @other.renew_invite!
    %w[dave erin].each do |name|
      get household_join_path(@other.invite_token)
      sign_in("#{name}@example.com")
      delete household_logout_path
    end
    assert_equal 3, @other.users.count
  end

  test "an expired or unknown link lets nobody in" do
    @other.update!(invite_token: "old-token", invite_expires_at: 1.minute.ago)
    [ "old-token", "made-up" ].each do |token|
      get household_join_path(token)
      assert_redirected_to household_login_path
    end
    sign_in("dave@example.com")
    assert_redirected_to household_login_path
    assert_not User.exists?(email: "dave@example.com")
  end

  test "someone already in a home stays there when they use another home's link" do
    @other.renew_invite!
    get household_join_path(@other.invite_token)
    sign_in("bob@example.com")
    assert_equal homes(:ours), users(:bob).reload.home
    assert_match "already in", flash[:notice]

    get household_join_path(@other.invite_token)
    assert_redirected_to household_root_path
    assert_equal homes(:ours), users(:bob).reload.home
  end

  test "a member can make and replace the invite link of their home" do
    sign_in("bob@example.com")
    post household_invite_path
    first = homes(:ours).reload.invite_token
    assert first.present?
    get household_members_path
    assert_includes response.body, first
    post household_invite_path
    assert_not_equal first, homes(:ours).reload.invite_token
    assert_nil Home.from_invite(first)
  end

  test "only a founder can start a household" do
    sign_in("bob@example.com")
    assert_no_difference -> { Home.count } do
      post household_homes_path, params: { home: { name: "Friends" } }
    end
    assert_response :not_found
    get household_members_path
    assert_select "h2", text: "Start another household", count: 0
  end

  test "a founder starts a household, gets its link and loses it once people join" do
    sign_in("alice@example.com")
    post household_homes_path, params: { home: { name: "Friends" } }
    friends = Home.find_by!(name: "Friends")
    assert_not friends.assistant?
    get household_members_path
    assert_includes response.body, friends.invite_token

    post household_invite_path, params: { home_id: friends.id }
    assert_redirected_to household_members_path

    friends.users.create!(email: "frank@example.com")
    post household_invite_path, params: { home_id: friends.id }
    assert_response :not_found
    get household_members_path
    assert_not_includes response.body, friends.reload.invite_token
    assert_equal homes(:ours), users(:alice).reload.home
  end

  test "a founder can't renew the link of a home someone else started" do
    sign_in("alice@example.com")
    post household_invite_path, params: { home_id: @other.id }
    assert_response :not_found
  end
end
