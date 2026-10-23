# One line in the household's activity log, read by the assistant through the API.
class Event < ApplicationRecord
  ACTIONS = %w[chore_added chore_edited chore_deleted chore_done chore_undone chore_claimed chore_released note_added transcript_added].freeze

  belongs_to :user, optional: true

  validates :action, inclusion: { in: ACTIONS }

  def self.log!(action, user: nil, by_claude: false, chore: nil, note: nil, summary: nil)
    create!(action: action, user: user, by_claude: by_claude, chore_id: chore&.id || note&.chore_id,
            note_id: note&.id, summary: (summary || chore&.title).to_s.truncate(250))
  end

  def as_api_json
    { id: id, action: action, at: created_at, actor: by_claude ? "claude" : user&.email,
      chore_id: chore_id, note_id: note_id, summary: summary }
  end
end
