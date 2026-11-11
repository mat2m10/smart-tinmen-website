# A list the whole home keeps together: groceries, gift ideas, a wish list.
# Items are ticked off as they are bought; unlike chores they earn no points.
class List < ApplicationRecord
  # Offered as one tap on an empty Lists page. Generic names only: what a
  # household actually keeps lives in the database, never in this code.
  SUGGESTED = [ "Groceries", "Gift ideas", "Wish list" ].freeze

  belongs_to :home
  has_many :items, -> { order(:created_at, :id) }, class_name: "ListItem", dependent: :delete_all, inverse_of: :list

  validates :name, presence: true, length: { maximum: 40 }, uniqueness: { scope: :home_id, case_sensitive: false }

  normalizes :name, with: ->(name) { name.squish }

  def self.visible_to(user)
    where(home_id: user.home_id)
  end
end
