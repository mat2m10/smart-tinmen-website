require "test_helper"

class Household::NotesControllerTest < ActionDispatch::IntegrationTest
  setup do
    @old = ENV["HOUSEHOLD_EMAILS"]
    ENV["HOUSEHOLD_EMAILS"] = "alice@example.com,bob@example.com"
    OmniAuth.config.test_mode = true
  end

  teardown do
    ENV["HOUSEHOLD_EMAILS"] = @old
    OmniAuth.config.mock_auth[:google_oauth2] = nil
    OmniAuth.config.test_mode = false
  end

  def sign_in(email)
    OmniAuth.config.mock_auth[:google_oauth2] = OmniAuth::AuthHash.new(provider: "google_oauth2", uid: email, info: { email: email, first_name: "Test" })
    get "/auth/google_oauth2/callback"
  end

  def upload(name, type)
    fixture_file_upload(name, type)
  end

  test "notes and attachments need a signed-in member" do
    note = Note.create!(body: "hello", user: users(:alice))
    get household_notes_path
    assert_redirected_to household_login_path
    post household_notes_path, params: { note: { body: "x" } }
    assert_redirected_to household_login_path
    attachment = note.attachments.create!(kind: "photo", content_type: "image/jpeg", byte_size: 3, data: "abc")
    get household_attachment_path(attachment)
    assert_redirected_to household_login_path
  end

  test "posts a note with a photo and a voice note, logs it, and serves the files back" do
    sign_in("alice@example.com")
    assert_difference -> { Event.where(action: "note_added").count }, 1 do
      post household_notes_path, params: { note: { body: "Out of soap", photos: [ upload("photo.jpg", "image/jpeg") ], voice: upload("voice.webm", "audio/webm") } }
    end
    assert_redirected_to household_notes_path
    note = Note.last
    assert_equal users(:alice), note.user
    assert_equal %w[photo voice], note.attachments.map(&:kind)
    assert_equal "image/jpeg", note.photos.first.content_type
    assert_equal "audio/webm", note.voices.first.content_type

    get household_attachment_path(note.photos.first)
    assert_response :success
    assert_equal "image/jpeg", response.media_type
    assert_includes response.headers["Cache-Control"], "private"

    get household_notes_path
    assert_select ".hh-note", 1
    assert_select ".hh-photos img", 1
    assert_select ".hh-note audio", 1
  end

  test "a note on a chore goes back to the chore and shows there" do
    sign_in("bob@example.com")
    chore = chores(:toilet)
    post household_notes_path, params: { note: { body: "Bleach is under the sink", chore_id: chore.id } }
    assert_redirected_to edit_household_chore_path(chore, anchor: "notes")
    get edit_household_chore_path(chore)
    assert_select ".hh-note-body", /Bleach is under the sink/
  end

  test "refuses a file that only pretends to be a photo, and empty notes" do
    sign_in("alice@example.com")
    assert_no_difference -> { Note.count } do
      post household_notes_path, params: { note: { body: "look", photos: [ upload("not_a_photo.jpg", "image/jpeg") ] } }
      post household_notes_path, params: { note: { body: " " } }
    end
    follow_redirect!
    assert_select ".hh-flash--alert"
  end

  test "members can delete their own and Claude's notes, not each other's" do
    sign_in("alice@example.com")
    mine = Note.create!(body: "mine", user: users(:alice))
    claudes = Note.create!(body: "from Claude", from_claude: true)
    bobs = Note.create!(body: "bob's", user: users(:bob))
    delete household_note_path(mine)
    delete household_note_path(claudes)
    delete household_note_path(bobs)
    assert_response :not_found
    assert_equal [ bobs ], Note.all.to_a
  end

  test "chore actions are logged for the assistant" do
    sign_in("alice@example.com")
    chore = chores(:toilet)
    post claim_household_chore_path(chore)
    post complete_household_chore_path(chore), params: { seen: chore.completions.count }
    assert_equal %w[chore_claimed chore_done], Event.order(:id).pluck(:action)
    assert_equal [ users(:alice).id ], Event.distinct.pluck(:user_id)
  end

  test "a chore's own notes text is untouched by its remarks" do
    sign_in("alice@example.com")
    chore = chores(:toilet)
    chore.update!(notes: "Use the blue brush")
    Note.create!(body: "remark", chore: chore, user: users(:alice))
    get edit_household_chore_path(chore)
    assert_select "textarea[name='chore[notes]']", text: "Use the blue brush"
    get household_root_path
    assert_select ".hh-note-count", /1/
  end

  test "refuses a file over the size limit without reading it" do
    sign_in("alice@example.com")
    big = Rack::Test::UploadedFile.new(StringIO.new("x" * (Attachment::MAX_BYTES["photo"] + 1)), "image/jpeg", original_filename: "big.jpg")
    assert_no_difference -> { Note.count } do
      post household_notes_path, params: { note: { body: "big", photos: [ big ] } }
    end
    follow_redirect!
    assert_select ".hh-flash--alert", /too large/
  end

  test "remarks on someone else's personal chore stay hidden" do
    chore = chores(:gp)
    chore.update!(personal: true, assignee: users(:bob))
    note = Note.create!(body: "Ask about the waiting list", chore: chore, user: users(:bob))
    sign_in("alice@example.com")
    get household_notes_path
    assert_select ".hh-note-body", text: /waiting list/, count: 0
    post household_notes_path, params: { note: { body: "peek", chore_id: chore.id } }
    assert_response :not_found
    assert_equal [ note ], chore.remarks.reload.to_a
  end

  test "deleting a personal chore takes its remarks and photos with it" do
    chore = chores(:gp)
    chore.update!(personal: true, assignee: users(:bob))
    sign_in("bob@example.com")
    post household_notes_path, params: { note: { body: "Passport scan", chore_id: chore.id, photos: [ upload("photo.jpg", "image/jpeg") ] } }
    photo = Note.last.photos.first

    sign_in("alice@example.com")
    get household_attachment_path(photo)
    assert_response :not_found

    sign_in("bob@example.com")
    delete household_chore_path(chore)
    assert_not Note.exists?(body: "Passport scan")
    assert_not Attachment.exists?(photo.id)
  end
end
