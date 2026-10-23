class AddNicknameToUsers < ActiveRecord::Migration[8.1]
  def change
    # The name a member picks on their profile; Google keeps filling "name".
    add_column :users, :nickname, :string
  end
end
