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
    assert_equal "livia@example.com", response.parsed_body.find { |c| c["title"] == "Find a GP" }["assignee_email"]
  end

  test "creates and updates a chore" do
    post household_api_chores_path, headers: auth, as: :json,
      params: { chore: { title: "Clean the oven", area: "cleaning", room: "Kitchen", due_on: Date.current, repeat_unit: "month", repeat_every: 3, effort: 3, assignee_email: "matthieu@example.com" } }
    assert_response :created
    chore = Chore.find_by!(title: "Clean the oven")
    assert_equal users(:matthieu), chore.assignee

    patch household_api_chore_path(chore), headers: auth, as: :json, params: { chore: { assignee_email: "", effort: 2 } }
    assert_response :success
    assert_nil chore.reload.assignee
    assert_equal 2, chore.effort
  end

  test "rejects an unknown assignee and invalid data" do
    post household_api_chores_path, headers: auth, as: :json, params: { chore: { title: "x", area: "cleaning", due_on: Date.current, assignee_email: "nobody@example.com" } }
    assert_response :unprocessable_entity
    post household_api_chores_path, headers: auth, as: :json, params: { chore: { title: "", area: "cleaning", due_on: Date.current } }
    assert_response :unprocessable_entity
  end

  test "lists completions since a date" do
    chores(:toilet).complete!(users(:livia))
    get household_api_completions_path(since: Date.current.iso8601), headers: auth
    assert_response :success
    assert_equal [ "livia@example.com" ], response.parsed_body.map { |c| c["user_email"] }
  end
end
