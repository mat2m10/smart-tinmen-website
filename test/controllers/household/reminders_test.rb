require "test_helper"

class Household::RemindersTest < ActionDispatch::IntegrationTest
  include ActiveJob::TestHelper

  KEY = "k" * 40
  TOKEN = "t" * 40

  setup do
    @env = ENV.to_h.slice("HOUSEHOLD_EMAILS", "HOUSEHOLD_CALENDAR_TOKEN", "HOUSEHOLD_API_TOKEN", "VAPID_PUBLIC_KEY", "VAPID_PRIVATE_KEY")
    ENV["HOUSEHOLD_EMAILS"] = "alice@example.com,bob@example.com"
    ENV["HOUSEHOLD_CALENDAR_TOKEN"] = KEY
    ENV["HOUSEHOLD_API_TOKEN"] = TOKEN
    keys = WebPush.generate_key
    ENV["VAPID_PUBLIC_KEY"] = keys.public_key
    ENV["VAPID_PRIVATE_KEY"] = keys.private_key
    OmniAuth.config.test_mode = true
  end

  teardown do
    %w[HOUSEHOLD_EMAILS HOUSEHOLD_CALENDAR_TOKEN HOUSEHOLD_API_TOKEN VAPID_PUBLIC_KEY VAPID_PRIVATE_KEY].each { |k| ENV[k] = @env[k] }
    OmniAuth.config.mock_auth[:google_oauth2] = nil
    OmniAuth.config.test_mode = false
  end

  def sign_in(email)
    OmniAuth.config.mock_auth[:google_oauth2] = OmniAuth::AuthHash.new(provider: "google_oauth2", uid: email, info: { email: email, first_name: "Test" })
    get "/auth/google_oauth2/callback"
  end

  def subscribe(user, endpoint = "https://push.example.com/#{user.id}")
    user.push_subscriptions.create!(endpoint: endpoint, p256dh: "p", auth: "a")
  end

  test "the front page has the note composer but lists notes only on the Notes page" do
    Note.create!(body: "Front page note", user: users(:bob))
    sign_in("alice@example.com")
    get household_root_path
    assert_select ".hh-front-notes form.hh-composer input[name=return_to][value=board]"
    assert_select ".hh-note-body", count: 0
    get household_notes_path
    assert_select ".hh-note-body", /Front page note/
    assert_enqueued_with(job: HouseholdPushJob, args: [ [ users(:bob).id ], "Test added a note", "from the board", household_notes_path ]) do
      post household_notes_path, params: { return_to: "board", note: { body: "from the board" } }
    end
    assert_redirected_to household_root_path
  end

  test "a new note notifies the other member, not the author" do
    sign_in("alice@example.com")
    assert_enqueued_with(job: HouseholdPushJob, args: [ [ users(:bob).id ], "Test added a note", "hello", household_notes_path ]) do
      post household_notes_path, params: { note: { body: "hello" } }
    end
  end

  test "members store and remove this phone's push subscription" do
    sign_in("alice@example.com")
    post household_push_subscription_path, as: :json, params: { subscription: { endpoint: "https://push.example.com/abc", keys: { p256dh: "p", auth: "a" } } }
    assert_response :created
    assert_equal users(:alice), PushSubscription.find_by!(endpoint: "https://push.example.com/abc").user
    delete household_push_subscription_path, as: :json, params: { endpoint: "https://push.example.com/abc" }
    assert_not PushSubscription.exists?(endpoint: "https://push.example.com/abc")
  end

  test "push subscriptions need a member and an https endpoint" do
    post household_push_subscription_path, as: :json, params: { subscription: { endpoint: "https://push.example.com/x", keys: { p256dh: "p", auth: "a" } } }
    assert_redirected_to household_login_path
    sign_in("alice@example.com")
    post household_push_subscription_path, as: :json, params: { subscription: { endpoint: "http://push.example.com/x", keys: { p256dh: "p", auth: "a" } } }
    assert_response :unprocessable_entity
  end

  test "the manifest and service worker are public and scoped to /household" do
    get household_manifest_path
    assert_response :success
    assert_equal "/household", JSON.parse(response.body)["scope"]
    get household_service_worker_path
    assert_response :success
    assert_equal "/household", response.headers["Service-Worker-Allowed"]
    assert_includes response.body, "showNotification"
  end

  test "Claude sends a reminder to one member through the API, and gone devices are removed" do
    subscribe(users(:alice))
    gone = subscribe(users(:alice), "https://push.example.com/gone")
    subscribe(users(:bob))
    sent = []
    send_push = lambda do |**args|
      raise WebPush::ExpiredSubscription.new(Struct.new(:body).new(""), "push.example.com") if args[:endpoint] == gone.endpoint

      sent << args[:endpoint]
    end
    original = WebPush.method(:payload_send)
    WebPush.define_singleton_method(:payload_send) { |**args| send_push.call(**args) }
    begin
      post household_api_push_path, headers: { "Authorization" => "Bearer #{TOKEN}" }, as: :json,
        params: { push: { to: "alice@example.com", title: "Today", body: "Pay the rent", path: "https://evil.example" } }
    ensure
      WebPush.define_singleton_method(:payload_send, original)
    end
    assert_response :success
    assert_equal 1, response.parsed_body["devices"]
    assert_equal [ "https://push.example.com/#{users(:alice).id}" ], sent
    assert_not PushSubscription.exists?(gone.id)
  end

  test "each push carries the recipient's badge count" do
    subscribe(users(:alice))
    subscribe(users(:bob))
    Chore.create!(title: "Bob's late chore", area: "Admin", due_on: Date.yesterday, assignee: users(:bob))
    badges = {}
    original = WebPush.method(:payload_send)
    WebPush.define_singleton_method(:payload_send) { |**args| badges[args[:endpoint]] = JSON.parse(args[:message])["badge"] }
    begin
      assert_equal 2, PushSubscription.notify(User.all, title: "Hi", body: "there")
    ensure
      WebPush.define_singleton_method(:payload_send, original)
    end
    assert_equal users(:alice).badge_count, badges["https://push.example.com/#{users(:alice).id}"]
    assert_equal users(:alice).badge_count + 1, badges["https://push.example.com/#{users(:bob).id}"]
  end

  test "pages set the app badge for the signed-in member" do
    sign_in("alice@example.com")
    get household_root_path
    assert_select "body[data-controller=badge][data-badge-count-value=?]", users(:alice).badge_count.to_s
  end

  test "the calendar feed needs the key and lists open chores as all-day events" do
    get household_calendar_path(key: "wrong")
    assert_response :not_found
    chores(:toilet).update!(notes: "Use gloves, the blue ones; thanks")
    get household_calendar_path(key: KEY)
    assert_response :success
    assert_equal "text/calendar", response.media_type
    body = response.body.gsub("\r\n ", "")
    assert_includes body, "SUMMARY:Clean the toilet"
    assert_includes body, "DTSTART;VALUE=DATE:#{chores(:toilet).due_on.strftime("%Y%m%d")}"
    assert_includes body, 'Use gloves\, the blue ones\; thanks'
    assert response.body.split("\r\n").all? { |line| line.bytesize <= 75 }

    get household_calendar_path(key: KEY, only: "planning")
    assert_includes response.body, "Find a GP"
    assert_not_includes response.body, "Clean the toilet"

    chores(:gp).update!(personal: true, assignee: users(:bob))
    get household_calendar_path(key: KEY)
    assert_not_includes response.body, "Find a GP"
  end

  test "the agenda shows two weeks day by day, with repeats ahead, all-day items before timed ones" do
    chores(:toilet).update!(due_on: Date.current)
    chores(:gp).update!(due_on: Date.current, due_time: "09:15")
    Chore.create!(title: "Old letter", area: "Admin", due_on: Date.current - 3)
    sign_in("alice@example.com")
    get household_agenda_path
    assert_response :success
    assert_select ".hh-group--overdue .hh-chore-title", "Old letter"
    assert_select ".hh-agenda-day", 14
    today = css_select(".hh-agenda-day").first
    assert_equal [ "Clean the toilet", "Find a GP" ], today.css(".hh-chore-title").map(&:text)
    assert_equal [ "All day", "09:15" ], today.css(".hh-agenda-time").map(&:text)
    assert_select ".hh-agenda-item--forecast .hh-chore-title", "Clean the toilet"
    Chore.create!(title: "Feed the cat", area: "Food", due_on: Date.current, repeat_unit: "day")
    get household_agenda_path
    assert_select ".hh-agenda-item .hh-chore-title", { text: "Feed the cat", count: 1 }

    get household_agenda_path(from: (Date.current + 7).iso8601)
    assert_select ".hh-group--overdue", 0
    get household_agenda_path(from: "not-a-date")
    assert_response :success
  end

  test "timed chores become one-hour events in the calendar feed, in UTC" do
    chores(:gp).update!(due_on: Date.new(2026, 11, 3), due_time: "14:30")
    get household_calendar_path(key: KEY)
    body = response.body.gsub("\r\n ", "")
    assert_includes body, "DTSTART:20261103T133000Z"
    assert_includes body, "DTEND:20261103T143000Z"
  end

  test "finishing or adding a chore tells the other member in one sentence" do
    sign_in("alice@example.com")
    chore = chores(:toilet)
    assert_enqueued_with(job: HouseholdPushJob, args: [ [ users(:bob).id ], "Test cleaned the toilet", "", household_root_path ]) do
      post complete_household_chore_path(chore), params: { seen: chore.completions.count }
    end
    assert_no_enqueued_jobs(only: HouseholdPushJob) do
      post complete_household_chore_path(chore), params: { seen: 0 } # stale double tap: already done
      post claim_household_chore_path(chore)
    end
    assert_enqueued_with(job: HouseholdPushJob, args: [ [ users(:bob).id ], "Test added “Buy milk”", "", household_root_path ]) do
      post household_chores_path, params: { chore: { title: "Buy milk", area: "Food", due_on: Date.current, repeat_unit: "none", repeat_every: 1, effort: 1 } }
    end
  end
end
