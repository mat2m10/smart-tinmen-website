class Household::Api::AttachmentsController < Household::Api::BaseController
  def show
    attachment = Attachment.find(params[:id])
    send_data attachment.data, type: attachment.content_type, disposition: "attachment",
              filename: "#{attachment.kind}-#{attachment.id}.#{Rack::Mime::MIME_TYPES.invert[attachment.content_type]&.delete(".") || "bin"}"
  end

  # Claude adds the text of a voice note so it shows under the recording.
  def update
    attachment = Attachment.without_data.find(params[:id])
    if attachment.update(transcript: params.expect(attachment: [ :transcript ])[:transcript].to_s.strip.presence)
      Event.log!("transcript_added", by_claude: true, note: attachment.note, summary: attachment.transcript)
      render json: attachment.as_api_json
    else
      render json: { errors: attachment.errors.full_messages }, status: :unprocessable_entity
    end
  end
end
