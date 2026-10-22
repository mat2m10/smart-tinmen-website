# Serves a stored photo or voice note to a signed-in household member.
class Household::AttachmentsController < Household::BaseController
  def show
    attachment = Attachment.where(note: Note.visible_to(current_user)).find(params[:id])
    response.headers["Cache-Control"] = "private, max-age=86400"
    send_data attachment.data, type: attachment.content_type, disposition: "inline"
  end
end
