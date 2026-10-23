# One time someone did a chore. Effort and planning are copied so the balance
# stays right when the chore is edited later.
class Completion < ApplicationRecord
  belongs_to :chore
  belongs_to :user, optional: true

  # The Done page lists this many days; older ticks stay for the balance.
  RECENT_DAYS = 21

  validates :completed_on, presence: true

  scope :recent, -> { where(completed_on: RECENT_DAYS.days.ago.to_date..) }
end
