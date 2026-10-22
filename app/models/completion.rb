# One time someone did a chore. Effort and planning are copied so the balance
# stays right when the chore is edited later.
class Completion < ApplicationRecord
  belongs_to :chore
  belongs_to :user, optional: true

  validates :completed_on, presence: true
end
