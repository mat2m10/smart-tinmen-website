require "net/http"

# Keeps one Google Calendar event per timed chore (an appointment) in a shared
# calendar. It signs in as a service account the calendar is shared with, so
# no person has to grant access. Without GOOGLE_SERVICE_ACCOUNT_JSON and
# GOOGLE_CALENDAR_ID it does nothing.
class Household::GoogleCalendar
  class Error < StandardError; end

  SCOPE = "https://www.googleapis.com/auth/calendar.events".freeze
  API = "https://www.googleapis.com/calendar/v3".freeze
  TIME_ZONE = "Europe/Amsterdam".freeze

  def self.from_env
    json, calendar_id = ENV["GOOGLE_SERVICE_ACCOUNT_JSON"].presence, ENV["GOOGLE_CALENDAR_ID"].presence
    new(credentials: JSON.parse(json), calendar_id: calendar_id) if json && calendar_id
  end

  # http is a callable (method, uri, headers, body) -> [status, body]; tests
  # pass their own instead of reaching Google.
  def initialize(credentials:, calendar_id:, http: method(:send_request))
    @credentials = credentials
    @calendar_id = calendar_id
    @http = http
  end

  # Event ids may only use a-v and 0-9, so "hhchore12" is valid.
  def self.event_id(chore_id)
    "hhchore#{chore_id}"
  end

  # Creates or moves the chore's event. PUT also revives an event that was
  # deleted before, which keeps its id taken; a missing event is inserted.
  def upsert(chore)
    event = event_for(chore)
    status, body = call(:put, event_path(chore.id), event)
    status, body = call(:post, "#{calendar_path}/events", event) if status == 404
    raise Error, "Calendar upsert for chore #{chore.id} failed (#{status}): #{body.to_s.truncate(200)}" unless status.between?(200, 299)
  end

  def delete(chore_id)
    status, body = call(:delete, event_path(chore_id))
    raise Error, "Calendar delete for chore #{chore_id} failed (#{status}): #{body.to_s.truncate(200)}" unless status.between?(200, 299) || status.in?([ 404, 410 ])
  end

  private

  def event_for(chore)
    start = Time.zone.parse("#{chore.due_on} #{chore.due_time}")
    owner = chore.assignee ? " (#{chore.assignee.display_name})" : ""
    details = [ chore.area, chore.room, chore.notes ].compact_blank.join(" · ")
    {
      id: self.class.event_id(chore.id), status: "confirmed",
      summary: chore.title + owner,
      description: [ details.presence, "https://smarttinmen.com/household/chores/#{chore.id}/edit" ].compact.join("\n\n"),
      start: { dateTime: start.iso8601, timeZone: TIME_ZONE },
      end: { dateTime: (start + 1.hour).iso8601, timeZone: TIME_ZONE }
    }
  end

  def calendar_path
    "/calendars/#{ERB::Util.url_encode(@calendar_id)}"
  end

  def event_path(chore_id)
    "#{calendar_path}/events/#{self.class.event_id(chore_id)}"
  end

  def call(verb, path, payload = nil)
    headers = { "Authorization" => "Bearer #{access_token}", "Content-Type" => "application/json" }
    @http.call(verb, URI("#{API}#{path}"), headers, payload&.to_json)
  end

  # A signed JWT is exchanged for an hour-long token, reused until near expiry.
  def access_token
    return @token if @token && Time.current < @token_expires_at

    now = Time.current.to_i
    claims = { iss: @credentials.fetch("client_email"), scope: SCOPE, aud: token_uri, iat: now, exp: now + 3600 }
    form = URI.encode_www_form(grant_type: "urn:ietf:params:oauth:grant-type:jwt-bearer", assertion: jwt(claims))
    status, body = @http.call(:post, URI(token_uri), { "Content-Type" => "application/x-www-form-urlencoded" }, form)
    raise Error, "Google token request failed (#{status})" unless status == 200

    data = JSON.parse(body)
    @token_expires_at = Time.current + data.fetch("expires_in", 3600).to_i - 60
    @token = data.fetch("access_token")
  end

  def token_uri
    @credentials["token_uri"].presence || "https://oauth2.googleapis.com/token"
  end

  def jwt(claims)
    segments = [ { alg: "RS256", typ: "JWT" }, claims ].map { |part| Base64.urlsafe_encode64(part.to_json, padding: false) }
    key = OpenSSL::PKey::RSA.new(@credentials.fetch("private_key"))
    signature = key.sign(OpenSSL::Digest.new("SHA256"), segments.join("."))
    [ *segments, Base64.urlsafe_encode64(signature, padding: false) ].join(".")
  end

  def send_request(verb, uri, headers, body)
    request = { get: Net::HTTP::Get, post: Net::HTTP::Post, put: Net::HTTP::Put, delete: Net::HTTP::Delete }.fetch(verb).new(uri, headers)
    request.body = body if body
    response = Net::HTTP.start(uri.host, uri.port, use_ssl: true, open_timeout: 5, read_timeout: 10) { |http| http.request(request) }
    [ response.code.to_i, response.body ]
  end
end
