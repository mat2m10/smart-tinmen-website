class Household::SessionsController < Household::BaseController
  skip_before_action :require_household_user

  def new
    redirect_to household_root_path if current_user
  end

  def create
    user = User.from_google(request.env["omniauth.auth"])
    if user
      reset_session
      session[:household_user_id] = user.id
      redirect_to household_root_path
    else
      redirect_to household_login_path, alert: "That Google account isn't on the household list."
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
