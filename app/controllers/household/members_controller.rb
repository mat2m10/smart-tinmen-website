# Who is in your household, the link to invite someone, and for the founders
# the other households they started.
class Household::MembersController < Household::BaseController
  def index
    @home = current_user.home
    @members = @home.users.order(:name, :email)
    @started_homes = current_user.started_homes.includes(:users).order(:name)
  end
end
