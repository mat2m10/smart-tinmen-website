# Loose notes for the household, and remarks on a chore, with photos and a voice note.
class Household::NotesController < Household::BaseController
  def index
    @notes = Note.visible_to(current_user).includes(:user, :chore, :attachments).recent_first.limit(50)
    @note = Note.new
  end

  def create
    @note = Note.new(body: note_params[:body], user: current_user)
    @note.chore = Chore.visible_to(current_user).find(note_params[:chore_id]) if note_params[:chore_id].present?
    Array(note_params[:photos]).compact_blank.first(Note::MAX_PHOTOS).each { |file| @note.attachments << Attachment.from_upload("photo", file) }
    @note.attachments << Attachment.from_upload("voice", note_params[:voice]) if note_params[:voice].present?

    if @note.attachments.all?(&:valid?) && @note.save
      summary = @note.body.presence || "(#{@note.attachments.map(&:kind).tally.map { |k, n| "#{n} #{k}" }.join(", ")})"
      Event.log!("note_added", user: current_user, note: @note, summary: summary)
      others = @note.chore ? @note.chore.audience(except: current_user) : User.where.not(id: current_user.id)
      # The board no longer lists notes, so the push opens where the note can be read.
      HouseholdPushJob.perform_later(others.ids, "#{current_user.display_name} added a note", summary, note_path) if others.exists?
      redirect_to back_path, notice: "Note added", status: :see_other
    else
      problems = (@note.errors.full_messages + @note.attachments.flat_map { |a| a.errors.full_messages }).uniq
      redirect_to back_path, alert: problems.to_sentence, status: :see_other
    end
  end

  def destroy
    notes = Note.visible_to(current_user)
    note = notes.where(user: current_user).or(notes.where(from_claude: true)).find(params[:id])
    note.destroy!
    redirect_to back_path(note.chore), notice: "Note deleted", status: :see_other
  end

  private

  def note_params
    params.fetch(:note, {}).permit(:body, :chore_id, :voice, photos: [])
  end

  def back_path(chore = @note&.chore)
    params[:return_to] == "board" ? household_root_path : note_path(chore)
  end

  def note_path(chore = @note&.chore)
    chore ? edit_household_chore_path(chore, anchor: "notes") : household_notes_path
  end
end
