# Until now the app held one household. Each group of people now gets a home of
# its own; everything that exists moves into the first one, which keeps the
# assistant and the other integrations configured through the environment.
class CreateHomes < ActiveRecord::Migration[8.1]
  TABLES = %i[users chores notes events].freeze

  def up
    create_table :homes do |t|
      t.string :name, null: false
      t.boolean :assistant, default: false, null: false
      t.string :invite_token
      t.datetime :invite_expires_at
      t.bigint :created_by_id
      t.timestamps
    end
    add_index :homes, :invite_token, unique: true
    add_index :homes, :created_by_id
    add_index :homes, :assistant, unique: true, where: "assistant", name: "index_homes_on_the_assistant_home"
    add_foreign_key :homes, :users, column: :created_by_id, on_delete: :nullify

    TABLES.each { |table| add_reference table, :home, foreign_key: true, index: true }

    if TABLES.any? { |table| select_value("SELECT 1 FROM #{table} LIMIT 1") }
      home_id = select_value("INSERT INTO homes (name, assistant, created_at, updated_at) VALUES ('Household', TRUE, NOW(), NOW()) RETURNING id")
      TABLES.each { |table| execute("UPDATE #{table} SET home_id = #{home_id.to_i}") }
    end

    # A user can be without a home (not invited yet); shared data never is.
    %i[chores notes events].each { |table| change_column_null table, :home_id, false }
  end

  def down
    TABLES.each { |table| remove_reference table, :home, foreign_key: true, index: true }
    drop_table :homes
  end
end
