# Starting another household, for someone new to the app. Only the founders
# (HOUSEHOLD_EMAILS) can, so the app stays invitation-only. They get a link to
# pass on and see nothing of what the new household adds.
class Household::HomesController < Household::BaseController
  def create
    return head(:not_found) unless current_user.founder?

    home = current_user.started_homes.new(params.expect(home: [ :name ]).merge(Home.fresh_invite))
    if home.save
      redirect_to household_members_path, notice: "#{home.name} is ready: send them the link", status: :see_other
    else
      redirect_to household_members_path, alert: home.errors.full_messages.to_sentence, status: :see_other
    end
  end
end
