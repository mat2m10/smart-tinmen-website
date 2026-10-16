class CreateHouseholdTables < ActiveRecord::Migration[8.1]
  def change
    create_table :users do |t|
      t.string :email, null: false
      t.string :name
      t.string :avatar_url
      t.timestamps
    end
    add_index :users, :email, unique: true

    create_table :chores do |t|
      t.string :title, null: false
      t.text :notes
      t.string :area, null: false, default: "home"
      t.string :room
      t.date :due_on, null: false
      t.string :repeat_unit, null: false, default: "none"
      t.integer :repeat_every, null: false, default: 1
      t.boolean :fixed_schedule, null: false, default: false
      t.integer :effort, null: false, default: 1
      t.boolean :planning, null: false, default: false
      t.references :assignee, foreign_key: { to_table: :users }
      t.references :created_by, foreign_key: { to_table: :users }
      t.datetime :done_at
      t.timestamps
    end
    add_index :chores, :due_on
    add_index :chores, :done_at

    create_table :completions do |t|
      t.references :chore, null: false, foreign_key: true
      t.references :user, foreign_key: true
      t.date :completed_on, null: false
      t.integer :effort, null: false, default: 1
      t.boolean :planning, null: false, default: false
      t.timestamps
    end
    add_index :completions, :completed_on
  end
end
