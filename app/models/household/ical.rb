# Builds an iCalendar (RFC 5545) document with one all-day event per chore,
# on its next due date.
class Household::Ical
  def initialize(chores, name:)
    @chores = chores
    @name = name
  end

  def to_s
    lines = [ "BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Smart Tinmen//Household//EN", "CALSCALE:GREGORIAN",
              "METHOD:PUBLISH", "X-WR-CALNAME:#{text(@name)}", "REFRESH-INTERVAL;VALUE=DURATION:PT1H" ]
    stamp = Time.current.utc.strftime("%Y%m%dT%H%M%SZ")
    @chores.each do |chore|
      owner = chore.assignee ? " (#{chore.assignee.display_name})" : ""
      details = [ chore.area, chore.room, chore.repeat_label, chore.effort_label, chore.notes ].compact_blank.join(" · ")
      lines += [ "BEGIN:VEVENT", "UID:chore-#{chore.id}@smarttinmen.com", "DTSTAMP:#{stamp}",
                 "DTSTART;VALUE=DATE:#{chore.due_on.strftime("%Y%m%d")}", "DTEND;VALUE=DATE:#{(chore.due_on + 1).strftime("%Y%m%d")}",
                 "SUMMARY:#{text(chore.title + owner)}", "DESCRIPTION:#{text(details)}",
                 "URL:https://smarttinmen.com/household/chores/#{chore.id}/edit", "TRANSP:TRANSPARENT", "END:VEVENT" ]
    end
    lines << "END:VCALENDAR"
    lines.map { |line| fold(line) }.join("\r\n") + "\r\n"
  end

  private

  def text(value)
    value.to_s.gsub("\\", "\\\\\\").gsub(";", "\\;").gsub(",", "\\,").gsub(/\r?\n/, "\\n")
  end

  # Lines longer than 75 octets continue on the next line after a space.
  def fold(line)
    parts = []
    current = +""
    line.each_char do |char|
      if (current.bytesize + char.bytesize) > (parts.empty? ? 75 : 74)
        parts << current
        current = +""
      end
      current << char
    end
    parts << current
    parts.join("\r\n ")
  end
end
