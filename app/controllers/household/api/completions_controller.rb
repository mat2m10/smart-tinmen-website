class Household::Api::CompletionsController < Household::Api::BaseController
  def index
    since = Date.iso8601(params[:since].to_s) rescue 90.days.ago.to_date
    completions = Completion.includes(:chore, :user).where(completed_on: since..).order(:completed_on)
    render json: completions.map { |c|
      { chore_id: c.chore_id, title: c.chore.title, area: c.chore.area, user_email: c.user&.email,
        completed_on: c.completed_on, effort: c.effort, planning: c.planning, personal: c.chore.personal }
    }
  end
end
