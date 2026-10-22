class Household::Api::ChoresController < Household::Api::BaseController
  FIELDS = %i[title notes area room due_on due_time repeat_unit repeat_every fixed_schedule effort planning assignee_email].freeze

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
