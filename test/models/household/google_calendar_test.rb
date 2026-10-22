require "test_helper"

class Household::GoogleCalendarTest < ActiveSupport::TestCase
  KEY = OpenSSL::PKey::RSA.new(2048)

  setup do
    @requests = []
    @responses = { put: [ 200, "{}" ], post: [ 200, "{}" ], delete: [ 204, "" ] }
    http = lambda do |verb, uri, headers, body|
      @requests << { verb: verb, uri: uri, headers: headers, body: body }
      next [ 200, { access_token: "tok", expires_in: 3600 }.to_json ] if uri.host == "oauth2.googleapis.com"

      @responses.fetch(verb)
    end
    credentials = { "client_email" => "household@example.iam.gserviceaccount.com", "private_key" => KEY.to_pem }
    @calendar = Household::GoogleCalendar.new(credentials: credentials, calendar_id: "abc@group.calendar.google.com", http: http)
    @chore = chores(:gp)
    @chore.update!(due_on: Date.new(2026, 11, 3), due_time: "14:30")
  end

  test "signs a JWT the service account's key verifies" do
    @calendar.upsert(@chore)
    token = @requests.first
    assert_equal "/token", token[:uri].path
    assertion = URI.decode_www_form(token[:body]).to_h.fetch("assertion")
    header, claims, signature = assertion.split(".")
    assert KEY.public_key.verify(OpenSSL::Digest.new("SHA256"), Base64.urlsafe_decode64(signature), "#{header}.#{claims}")
    payload = JSON.parse(Base64.urlsafe_decode64(claims))
    assert_equal Household::GoogleCalendar::SCOPE, payload["scope"]
    assert_equal "household@example.iam.gserviceaccount.com", payload["iss"]
  end

  test "puts a one-hour event in Amsterdam time with a fixed id" do
    @calendar.upsert(@chore)
    put = @requests.last
    assert_equal :put, put[:verb]
    assert_equal "/calendar/v3/calendars/abc%40group.calendar.google.com/events/hhchore#{@chore.id}", put[:uri].path
    assert_equal "Bearer tok", put[:headers]["Authorization"]
    event = JSON.parse(put[:body])
    assert_equal "Find a GP (Bob)", event["summary"]
    assert_equal "2026-11-03T14:30:00+01:00", event["start"]["dateTime"]
    assert_equal "2026-11-03T15:30:00+01:00", event["end"]["dateTime"]
    assert_equal "Europe/Amsterdam", event["start"]["timeZone"]
    assert_match %r{/household/chores/#{@chore.id}/edit}, event["description"]
  end

  test "inserts the event when it does not exist yet" do
    @responses[:put] = [ 404, "{}" ]
    @calendar.upsert(@chore)
    assert_equal :post, @requests.last[:verb]
    assert_equal "hhchore#{@chore.id}", JSON.parse(@requests.last[:body])["id"]
  end

  test "reuses the access token" do
    @calendar.upsert(@chore)
    @calendar.delete(@chore.id)
    assert_equal 1, @requests.count { |request| request[:uri].host == "oauth2.googleapis.com" }
  end

  test "deleting an event that is already gone is fine" do
    @responses[:delete] = [ 410, "" ]
    assert_nothing_raised { @calendar.delete(@chore.id) }
  end

  test "other failures raise so the job reports them" do
    @responses[:put] = [ 403, "forbidden" ]
    assert_raises(Household::GoogleCalendar::Error) { @calendar.upsert(@chore) }
  end

  test "is off without credentials" do
    assert_nil Household::GoogleCalendar.from_env
  end
end
