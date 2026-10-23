# Serves a member's profile picture to a signed-in household member. The URL
# carries the profile's last change, so it can be cached for long.
class Household::AvatarsController < Household::BaseController
  def show
    user = User.where.not(photo_type: nil).find(params[:id])
    response.headers["Cache-Control"] = "private, max-age=31536000"
    send_data user.photo, type: user.photo_type, disposition: "inline"
  end
end
