class Household::Api::EventsController < Household::Api::BaseController
  LIMIT = 200

  # Everything that happened after event `after` (an id), oldest first.
  def index
    events = Event.includes(:user).where(id: (params[:after].to_i + 1)..).order(:id).limit(LIMIT)
    render json: events.map(&:as_api_json)
  end
end
