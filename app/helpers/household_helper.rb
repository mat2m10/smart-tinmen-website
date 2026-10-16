module HouseholdHelper
  def due_label(chore, today = Date.current)
    days = (chore.due_on - today).to_i
    if days < -1 then "#{-days} days late"
    elsif days == -1 then "Yesterday"
    elsif days.zero? then "Today"
    elsif days == 1 then "Tomorrow"
    elsif days < 7 then chore.due_on.strftime("%A")
    else chore.due_on.strftime("%-d %b %Y")
    end
  end

  def claim_label(chore)
    if chore.assignee == current_user then "Drop"
    elsif chore.assignee then "Take over"
    else "I'll do it"
    end
  end
end
