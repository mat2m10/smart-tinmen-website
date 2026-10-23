# Everything under /household needs a signed-in household member and stays out of search engines.
class Household::BaseController < ApplicationController
  layout "household"
  before_action :keep_out_of_search
  before_action :require_household_user
  helper_method :current_user

  private

  def current_user
    return @current_user if defined?(@current_user)

    user = User.find_by(id: session[:household_user_id])
    @current_user = user if user && User.allowed?(user.email)
  end

  # Remembers the page so signing in leads back to it (the tabletop app shares this login).
  def require_household_user
    return if current_user

    session[:return_to] = request.fullpath if request.get? || request.head?
    redirect_to household_login_path
  end

  def keep_out_of_search
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
  end
end
