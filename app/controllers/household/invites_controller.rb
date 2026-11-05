# Invite links. Opening one remembers the invite for the Google sign-in that
# follows; the link stays valid for a while, so housemates can use the same one.
class Household::InvitesController < Household::BaseController
  skip_before_action :require_household_user, only: :show

  def show
    @home = Home.from_invite(params[:token])
    return redirect_to(household_login_path, alert: "That invite link has expired. Ask for a new one.") unless @home

    if current_user
      notice = "You are already in #{current_user.home.name}." unless current_user.home == @home
      return redirect_to(household_root_path, notice: notice)
    end
    session[:home_invite] = params[:token]
  end

  # A new link for your own home, or for one you started that nobody joined yet.
  # Once people live there it is theirs alone.
  def create
    home = params[:home_id].present? ? current_user.started_homes.where.missing(:users).find(params[:home_id]) : current_user.home
    home.renew_invite!
    redirect_to household_members_path, notice: "New invite link ready for #{home.name}", status: :see_other
  end
end
