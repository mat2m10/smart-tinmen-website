class CreateHouseholdNotes < ActiveRecord::Migration[8.1]
  def change
    create_table :notes do |t|
      t.references :chore, foreign_key: { on_delete: :nullify }
      t.references :user, foreign_key: { on_delete: :nullify }
      t.boolean :from_claude, null: false, default: false
      t.text :body
      t.timestamps
    end
    add_index :notes, :created_at

    # Photos and voice notes live in Postgres: a household makes few and small
    # ones, and it keeps them behind the same login as everything else.
    create_table :attachments do |t|
      t.references :note, null: false, foreign_key: { on_delete: :cascade }
      t.string :kind, null: false
      t.string :content_type, null: false
      t.integer :byte_size, null: false
      t.binary :data, null: false
      t.text :transcript
      t.timestamps
    end

    # What happened, in order, so an assistant can catch up with "since event N".
    create_table :events do |t|
      t.string :action, null: false
      t.references :user, foreign_key: { on_delete: :nullify }
      t.boolean :by_claude, null: false, default: false
      t.bigint :chore_id
      t.bigint :note_id
      t.string :summary
      t.datetime :created_at, null: false
    end
  end
end
