class Household::Api::ChoresController < Household::Api::BaseController
  FIELDS = %i[title notes area room due_on due_time repeat_unit repeat_every fixed_schedule effort planning personal assignee_email].freeze

  def index
    chores = params[:include_done] == "true" ? Chore.all : Chore.pending
    render json: chores.includes(:assignee, :completions).order(:due_on, :title).map(&:as_api_json)
  end

  def create
    save(Chore.new, :created)
  end

  def update
    save(Chore.find(params[:id]), :ok)
  end

  # Ticks a chore off for whoever said they did it ("I did the laundry"), on
  # the day they did it. A one-off chore that is already done counts once.
  def complete
    chore = Chore.find(params[:id])
    attrs = params.expect(completion: %i[by on])
    doer = User.find_by(email: attrs[:by].to_s.strip.downcase)
    return render json: { errors: [ "Unknown by" ] }, status: :unprocessable_entity unless doer

    on = attrs[:on].present? ? (Date.iso8601(attrs[:on]) rescue nil) : Date.current
    return render json: { errors: [ "on must be a date, not in the future" ] }, status: :unprocessable_entity if on.nil? || on > Date.current

    done = chore.with_lock do
      next false if chore.done_at?

      chore.complete!(doer, on: on)
      Event.log!("chore_done", by_claude: true, chore: chore, summary: "#{chore.title} (done by #{doer.display_name})")
      true
    end
    return render json: { errors: [ "Already done" ] }, status: :unprocessable_entity unless done

    render json: chore.reload.as_api_json
  end

  private

  def save(chore, status)
    attrs = params.expect(chore: FIELDS).to_h
    if attrs.key?("assignee_email")
      email = attrs.delete("assignee_email")
      chore.assignee = email.present? ? User.find_by(email: email.strip.downcase) : nil
      return render json: { errors: [ "Unknown assignee_email" ] }, status: :unprocessable_entity if email.present? && chore.assignee.nil?
    end

    if chore.update(attrs)
      Event.log!(status == :created ? "chore_added" : "chore_edited", by_claude: true, chore: chore)
      render json: chore.as_api_json, status: status
    else
      render json: { errors: chore.errors.full_messages }, status: :unprocessable_entity
    end
  end
end
