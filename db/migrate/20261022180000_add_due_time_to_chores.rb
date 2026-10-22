class AddDueTimeToChores < ActiveRecord::Migration[8.1]
  def change
    add_column :chores, :due_time, :string, limit: 5
  end
end
