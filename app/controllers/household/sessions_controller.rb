class Household::SessionsController < Household::BaseController
  skip_before_action :require_household_user

  def new
    redirect_to household_root_path if current_user
  end

  def create
    invite = Home.from_invite(session[:home_invite])
    user = User.from_google(request.env["omniauth.auth"], invite: invite)
    if user
      reset_session
      session[:household_user_id] = user.id
      notice = "You are already in #{user.home.name}, so the invite wasn't used." if invite && user.home != invite
      redirect_to household_root_path, notice: notice
    else
      redirect_to household_login_path, alert: "That Google account isn't in a household yet. Ask a member for an invite link."
    end
  end

  def failure
    redirect_to household_login_path, alert: "Google sign-in didn't work. Please try again."
  end

  def destroy
    reset_session
    redirect_to household_login_path, notice: "Signed out.", status: :see_other
  end
end
