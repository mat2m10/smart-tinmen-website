require "test_helper"

class Household::ApiTest < ActionDispatch::IntegrationTest
  TOKEN = "t" * 40

  setup { @old = ENV["HOUSEHOLD_API_TOKEN"]; ENV["HOUSEHOLD_API_TOKEN"] = TOKEN }
  teardown { ENV["HOUSEHOLD_API_TOKEN"] = @old }

  def auth(token = TOKEN)
    { "Authorization" => "Bearer #{token}" }
  end

  test "refuses requests without the right token" do
    get household_api_chores_path
    assert_response :unauthorized
    get household_api_chores_path, headers: auth("wrong")
    assert_response :unauthorized
  end

  test "refuses everything when the configured token is too short" do
    ENV["HOUSEHOLD_API_TOKEN"] = "short"
    get household_api_chores_path, headers: auth("short")
    assert_response :unauthorized
  end

  test "lists open chores" do
    get household_api_chores_path, headers: auth
    assert_response :success
    titles = response.parsed_body.map { |c| c["title"] }
    assert_includes titles, "Clean the toilet"
    assert_equal "bob@example.com", response.parsed_body.find { |c| c["title"] == "Find a GP" }["assignee_email"]
  end

  test "creates and updates a chore" do
    post household_api_chores_path, headers: auth, as: :json,
      params: { chore: { title: "Clean the oven", area: "Cleaning", room: "Kitchen", due_on: Date.current, repeat_unit: "month", repeat_every: 3, effort: 3, assignee_email: "alice@example.com" } }
    assert_response :created
    chore = Chore.find_by!(title: "Clean the oven")
    assert_equal users(:alice), chore.assignee

    patch household_api_chore_path(chore), headers: auth, as: :json, params: { chore: { assignee_email: "", effort: 2, due_time: "10:00" } }
    assert_response :success
    assert_nil chore.reload.assignee
    assert_equal 2, chore.effort
    assert_equal "10:00", response.parsed_body["due_time"]
  end

  test "rejects an unknown assignee and invalid data" do
    post household_api_chores_path, headers: auth, as: :json, params: { chore: { title: "x", area: "Cleaning", due_on: Date.current, assignee_email: "nobody@example.com" } }
    assert_response :unprocessable_entity
    post household_api_chores_path, headers: auth, as: :json, params: { chore: { title: "", area: "Cleaning", due_on: Date.current } }
    assert_response :unprocessable_entity
  end

  test "lists completions since a date" do
    chores(:toilet).complete!(users(:bob))
    get household_api_completions_path(since: Date.current.iso8601), headers: auth
    assert_response :success
    assert_equal [ "bob@example.com" ], response.parsed_body.map { |c| c["user_email"] }
  end

  test "lists events after a given id" do
    first = Event.log!("chore_done", user: users(:alice), chore: chores(:toilet))
    second = Event.log!("note_added", user: users(:bob), summary: "Out of milk")
    get household_api_events_path(after: first.id), headers: auth
    assert_response :success
    assert_equal [ second.id ], response.parsed_body.map { |e| e["id"] }
    assert_equal "bob@example.com", response.parsed_body.first["actor"]
  end

  test "Claude posts a note, reads notes with their files, and adds a transcript" do
    post household_api_notes_path, headers: auth, as: :json, params: { note: { body: "Added 'buy milk' for Saturday", chore_id: chores(:toilet).id } }
    assert_response :created
    assert Note.last.from_claude
    assert Event.last.by_claude

    voice = Note.create!(user: users(:alice), body: "listen").attachments.create!(kind: "voice", content_type: "audio/webm", byte_size: 4, data: "abcd")
    get household_api_notes_path(ids: voice.note_id), headers: auth
    assert_equal "voice", response.parsed_body.first["attachments"].first["kind"]

    get household_api_attachment_path(voice), headers: auth
    assert_response :success
    assert_equal "abcd", response.body

    patch household_api_attachment_path(voice), headers: auth, as: :json, params: { attachment: { transcript: "We need milk" } }
    assert_response :success
    assert_equal "We need milk", voice.reload.transcript
    assert_equal "transcript_added", Event.last.action
  end

  test "Claude can't attach a note to a chore that doesn't exist" do
    post household_api_notes_path, headers: auth, as: :json, params: { note: { body: "hi", chore_id: 0 } }
    assert_response :unprocessable_entity
  end

  test "chores changed through the API are logged as Claude" do
    patch household_api_chore_path(chores(:toilet)), headers: auth, as: :json, params: { chore: { effort: 2 } }
    assert_equal [ true ], Event.pluck(:by_claude)
  end

  test "Claude ticks a chore off for whoever did it, on the day they did it" do
    chore = chores(:toilet)
    post complete_household_api_chore_path(chore), headers: auth, as: :json, params: { completion: { by: "Alice@example.com ", on: Date.yesterday.iso8601 } }
    assert_response :success
    completion = chore.completions.last
    assert_equal [ users(:alice), Date.yesterday ], [ completion.user, completion.completed_on ]
    assert_equal chore.reload.due_on.iso8601, response.parsed_body["due_on"]
    event = Event.last
    assert_equal [ "chore_done", true, "Clean the toilet (done by Alice)" ], [ event.action, event.by_claude, event.summary ]
  end

  test "a one-off chore done through the API counts once" do
    chore = chores(:gp)
    post complete_household_api_chore_path(chore), headers: auth, as: :json, params: { completion: { by: "bob@example.com" } }
    assert_response :success
    assert chore.reload.done_at
    assert_equal Date.current, chore.completions.last.completed_on
    assert_no_difference "Completion.count" do
      post complete_household_api_chore_path(chore), headers: auth, as: :json, params: { completion: { by: "bob@example.com" } }
    end
    assert_response :unprocessable_entity
  end

  test "completing needs a known member and a date that is not in the future" do
    chore = chores(:toilet)
    assert_no_difference "Completion.count" do
      post complete_household_api_chore_path(chore), headers: auth, as: :json, params: { completion: { by: "nobody@example.com" } }
      assert_response :unprocessable_entity
      post complete_household_api_chore_path(chore), headers: auth, as: :json, params: { completion: { by: "alice@example.com", on: Date.tomorrow.iso8601 } }
      assert_response :unprocessable_entity
      post complete_household_api_chore_path(chore), headers: auth, as: :json, params: { completion: { by: "alice@example.com", on: "soon" } }
      assert_response :unprocessable_entity
    end
  end
end
