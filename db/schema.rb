# This file is auto-generated from the current state of the database. Instead
# of editing this file, please use the migrations feature of Active Record to
# incrementally modify your database, and then regenerate this schema definition.
#
# This file is the source Rails uses to define your schema when running `bin/rails
# db:schema:load`. When creating a new database, `bin/rails db:schema:load` tends to
# be faster and is potentially less error prone than running all of your
# migrations from scratch. Old migrations may fail to apply correctly if those
# migrations use external dependencies or application code.
#
# It's strongly recommended that you check this file into your version control system.

ActiveRecord::Schema[8.1].define(version: 2026_10_22_120000) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"

  create_table "attachments", force: :cascade do |t|
    t.bigint "note_id", null: false
    t.string "kind", null: false
    t.string "content_type", null: false
    t.integer "byte_size", null: false
    t.binary "data", null: false
    t.text "transcript"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["note_id"], name: "index_attachments_on_note_id"
  end

  create_table "chores", force: :cascade do |t|
    t.string "area", default: "Home & repairs", null: false
    t.bigint "assignee_id"
    t.datetime "created_at", null: false
    t.bigint "created_by_id"
    t.datetime "done_at"
    t.date "due_on", null: false
    t.integer "effort", default: 1, null: false
    t.boolean "fixed_schedule", default: false, null: false
    t.text "notes"
    t.boolean "planning", default: false, null: false
    t.integer "repeat_every", default: 1, null: false
    t.string "repeat_unit", default: "none", null: false
    t.string "room"
    t.string "title", null: false
    t.datetime "updated_at", null: false
    t.index ["assignee_id"], name: "index_chores_on_assignee_id"
    t.index ["created_by_id"], name: "index_chores_on_created_by_id"
    t.index ["done_at"], name: "index_chores_on_done_at"
    t.index ["due_on"], name: "index_chores_on_due_on"
  end

  create_table "completions", force: :cascade do |t|
    t.bigint "chore_id", null: false
    t.date "completed_on", null: false
    t.datetime "created_at", null: false
    t.integer "effort", default: 1, null: false
    t.boolean "planning", default: false, null: false
    t.datetime "updated_at", null: false
    t.bigint "user_id"
    t.index ["chore_id"], name: "index_completions_on_chore_id"
    t.index ["completed_on"], name: "index_completions_on_completed_on"
    t.index ["user_id"], name: "index_completions_on_user_id"
  end

  create_table "events", force: :cascade do |t|
    t.string "action", null: false
    t.bigint "user_id"
    t.boolean "by_claude", default: false, null: false
    t.bigint "chore_id"
    t.bigint "note_id"
    t.string "summary"
    t.datetime "created_at", null: false
    t.index ["user_id"], name: "index_events_on_user_id"
  end

  create_table "notes", force: :cascade do |t|
    t.bigint "chore_id"
    t.bigint "user_id"
    t.boolean "from_claude", default: false, null: false
    t.text "body"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["chore_id"], name: "index_notes_on_chore_id"
    t.index ["created_at"], name: "index_notes_on_created_at"
    t.index ["user_id"], name: "index_notes_on_user_id"
  end

  create_table "users", force: :cascade do |t|
    t.string "avatar_url"
    t.datetime "created_at", null: false
    t.string "email", null: false
    t.string "name"
    t.datetime "updated_at", null: false
    t.index ["email"], name: "index_users_on_email", unique: true
  end

  add_foreign_key "attachments", "notes", on_delete: :cascade
  add_foreign_key "chores", "users", column: "assignee_id"
  add_foreign_key "chores", "users", column: "created_by_id"
  add_foreign_key "completions", "chores"
  add_foreign_key "completions", "users"
  add_foreign_key "events", "users", on_delete: :nullify
  add_foreign_key "notes", "chores", on_delete: :nullify
  add_foreign_key "notes", "users", on_delete: :nullify
end
