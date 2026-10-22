class AddPersonalToChores < ActiveRecord::Migration[8.1]
  def change
    add_column :chores, :personal, :boolean, default: false, null: false
  end
end
