class AddColorAndUndoToHousehold < ActiveRecord::Migration[8.1]
  def change
    # Each member's own colour ("#3F8F5A"), used for their chores and avatar.
    add_column :users, :color, :string
    # The due date before this completion moved it, so the tick can be undone.
    add_column :completions, :previous_due_on, :date
  end
end
