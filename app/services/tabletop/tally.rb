module Tabletop
  # A count shown in the list totals, with the range the rules allow.
  Tally = Struct.new(:label, :count, :min, :max) do
    def ok?
      count >= min.to_i && (max.nil? || count <= max)
    end

    def range
      if max.nil? then "#{min}+"
      elsif min.to_i == max then max.to_s
      else "#{min.to_i}–#{max}"
      end
    end
  end
end
