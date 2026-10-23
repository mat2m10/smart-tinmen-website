class AddPhotoToUsers < ActiveRecord::Migration[8.1]
  def change
    # Profile picture, a small square JPEG made on the phone (about 20 kB).
    add_column :users, :photo, :binary
    add_column :users, :photo_type, :string
  end
end
