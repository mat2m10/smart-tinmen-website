# Your own profile: the name and colour the household sees. It belongs to the
# site account (User), so other apps on the site can show it too.
class Household::ProfilesController < Household::BaseController
  def show
  end

  def update
    if current_user.update(params.expect(user: %i[nickname color]))
      redirect_to household_profile_path, notice: "Profile saved", status: :see_other
    else
      render :show, status: :unprocessable_entity
    end
  end
end
