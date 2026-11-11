# Shared lists for each home: groceries to buy, gift ideas, a wish list.
# Columns are in alphabetical order, the order db/schema.rb lists them in.
class CreateLists < ActiveRecord::Migration[8.1]
  def change
    create_table :lists do |t|
      t.datetime :created_at, null: false
      t.references :home, null: false, foreign_key: true
      t.string :name, null: false
      t.datetime :updated_at, null: false
    end

    create_table :list_items do |t|
      t.references :added_by, foreign_key: { to_table: :users, on_delete: :nullify }
      t.string :body, null: false
      t.datetime :checked_at
      t.datetime :created_at, null: false
      t.references :list, null: false, foreign_key: { on_delete: :cascade }
      t.datetime :updated_at, null: false
    end
  end
end
