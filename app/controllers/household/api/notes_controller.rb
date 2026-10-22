class Household::Api::NotesController < Household::Api::BaseController
  def index
    notes = Note.includes(:user, :chore, :attachments).recent_first.limit(100)
    notes = notes.where(id: params[:ids].to_s.split(",")) if params[:ids].present?
    notes = notes.where(chore_id: params[:chore_id]) if params[:chore_id].present?
    render json: notes.map(&:as_api_json)
  end

  # A note written by Claude, shown in the app with Claude as the author.
  def create
    attrs = params.expect(note: %i[body chore_id])
    note = Note.new(body: attrs[:body], from_claude: true)
    if attrs[:chore_id].present?
      note.chore = Chore.find_by(id: attrs[:chore_id])
      return render json: { errors: [ "Unknown chore_id" ] }, status: :unprocessable_entity unless note.chore
    end

    if note.save
      Event.log!("note_added", by_claude: true, note: note, summary: note.body)
      render json: note.as_api_json, status: :created
    else
      render json: { errors: note.errors.full_messages }, status: :unprocessable_entity
    end
  end
end
