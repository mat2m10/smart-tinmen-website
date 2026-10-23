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

ActiveRecord::Schema[8.1].define(version: 2026_10_23_200000) do
  # These are extensions that must be enabled in order to support this database
  enable_extension "pg_catalog.plpgsql"

  create_table "army_list_entries", force: :cascade do |t|
    t.bigint "army_list_id", null: false
    t.bigint "game_unit_id", null: false
    t.integer "position", default: 0, null: false
    t.jsonb "upgrade_ids", default: [], null: false
    t.string "loadout"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["army_list_id"], name: "index_army_list_entries_on_army_list_id"
    t.index ["game_unit_id"], name: "index_army_list_entries_on_game_unit_id"
  end

  create_table "army_lists", force: :cascade do |t|
    t.bigint "user_id", null: false
    t.string "game", null: false
    t.string "faction", null: false
    t.string "name", null: false
    t.integer "points_limit"
    t.integer "ploys", default: 0, null: false
    t.text "notes"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["user_id"], name: "index_army_lists_on_user_id"
  end

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
    t.string "due_time", limit: 5
    t.boolean "personal", default: false, null: false
    t.index ["assignee_id"], name: "index_chores_on_assignee_id"
    t.index ["created_by_id"], name: "index_chores_on_created_by_id"
    t.index ["done_at"], name: "index_chores_on_done_at"
    t.index ["due_on"], name: "index_chores_on_due_on"
  end

  create_table "collection_items", force: :cascade do |t|
    t.bigint "user_id", null: false
    t.string "item_type", null: false
    t.bigint "item_id", null: false
    t.integer "built", default: 0, null: false
    t.integer "painted", default: 0, null: false
    t.integer "unbuilt", default: 0, null: false
    t.string "note"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["user_id", "item_type", "item_id"], name: "index_collection_items_on_user_id_and_item_type_and_item_id", unique: true
    t.index ["user_id"], name: "index_collection_items_on_user_id"
  end

  create_table "completions", force: :cascade do |t|
    t.bigint "chore_id", null: false
    t.date "completed_on", null: false
    t.datetime "created_at", null: false
    t.integer "effort", default: 1, null: false
    t.boolean "planning", default: false, null: false
    t.date "previous_due_on"
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

  create_table "game_factions", force: :cascade do |t|
    t.string "game", null: false
    t.string "name", null: false
    t.jsonb "rules", default: {}, null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["game", "name"], name: "index_game_factions_on_game_and_name", unique: true
  end

  create_table "game_units", force: :cascade do |t|
    t.string "game", null: false
    t.string "faction", null: false
    t.string "name", null: false
    t.string "subtitle"
    t.string "category", null: false
    t.integer "points"
    t.integer "points_printed"
    t.integer "minis", default: 1, null: false
    t.boolean "unique", default: false, null: false
    t.jsonb "slots", default: [], null: false
    t.jsonb "keywords", default: [], null: false
    t.jsonb "stats", default: {}, null: false
    t.integer "selection_cost"
    t.text "notes"
    t.boolean "check", default: false, null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["game", "faction", "name"], name: "index_game_units_on_game_and_faction_and_name", unique: true
  end

  create_table "game_upgrades", force: :cascade do |t|
    t.string "game", null: false
    t.string "faction", null: false
    t.string "name", null: false
    t.string "slot", null: false
    t.integer "points"
    t.integer "points_printed"
    t.string "restriction"
    t.boolean "unique", default: false, null: false
    t.integer "limited"
    t.boolean "adds_mini", default: false, null: false
    t.integer "minis_added"
    t.boolean "removed", default: false, null: false
    t.text "notes"
    t.boolean "check", default: false, null: false
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["game", "faction", "name"], name: "index_game_upgrades_on_game_and_faction_and_name", unique: true
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

  create_table "push_subscriptions", force: :cascade do |t|
    t.bigint "user_id", null: false
    t.string "endpoint", null: false
    t.string "p256dh", null: false
    t.string "auth", null: false
    t.string "user_agent"
    t.datetime "created_at", null: false
    t.datetime "updated_at", null: false
    t.index ["endpoint"], name: "index_push_subscriptions_on_endpoint", unique: true
    t.index ["user_id"], name: "index_push_subscriptions_on_user_id"
  end

  create_table "users", force: :cascade do |t|
    t.string "avatar_url"
    t.string "color"
    t.datetime "created_at", null: false
    t.string "email", null: false
    t.string "name"
    t.string "nickname"
    t.binary "photo"
    t.string "photo_type"
    t.datetime "updated_at", null: false
    t.index ["email"], name: "index_users_on_email", unique: true
  end

  add_foreign_key "army_list_entries", "army_lists", on_delete: :cascade
  add_foreign_key "army_list_entries", "game_units"
  add_foreign_key "army_lists", "users", on_delete: :cascade
  add_foreign_key "attachments", "notes", on_delete: :cascade
  add_foreign_key "chores", "users", column: "assignee_id"
  add_foreign_key "chores", "users", column: "created_by_id"
  add_foreign_key "collection_items", "users", on_delete: :cascade
  add_foreign_key "completions", "chores"
  add_foreign_key "completions", "users"
  add_foreign_key "events", "users", on_delete: :nullify
  add_foreign_key "notes", "chores", on_delete: :nullify
  add_foreign_key "notes", "users", on_delete: :nullify
  add_foreign_key "push_subscriptions", "users", on_delete: :cascade
end
