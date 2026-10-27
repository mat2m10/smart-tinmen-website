# The Done page: what was ticked off in the last three weeks, so a wrong or
# test tick can be taken back or put on the right person.
class Household::CompletionsController < Household::BaseController
  before_action :set_completion, only: %i[update destroy]

  def index
    @completions = Completion.recent.where(chore: Chore.visible_to(current_user))
                             .includes(:user, chore: %i[assignee completions]).order(completed_on: :desc, created_at: :desc)
    @users = User.order(:name, :email)
  end

  # Only who did it can change; the date and points stay as they were.
  # A personal tick stays with its owner.
  def update
    return head(:not_found) if @completion.chore.personal

    @completion.update!(user: User.find(params.expect(completion: [ :user_id ])[:user_id]))
    redirect_to household_completions_path, notice: "Now on #{@completion.user.display_name}: #{@completion.chore.title}", status: :see_other
  end

  def destroy
    return redirect_to(back_path, notice: "That tick was already taken back", status: :see_other) unless @completion

    chore = @completion.chore
    back_on_list = chore.undo!(@completion)
    Event.log!("chore_undone", user: current_user, chore: chore)
    notice = back_on_list ? "Not done: #{chore.title}. It is back on the list" : "Removed an older tick of #{chore.title}; its points are off the counter"
    redirect_to back_path, notice: notice, status: :see_other
  end

  private

  # A second tap on Undo, or a tick someone else already took back, no longer
  # exists: destroy says so instead of showing a 404. Someone else's personal
  # tick still exists and stays a 404.
  def set_completion
    @completion = Completion.where(chore: Chore.visible_to(current_user)).find_by(id: params[:id])
    head(:not_found) if @completion.nil? && (action_name != "destroy" || Completion.exists?(params[:id]))
  end

  # Undo on the board's "Done" message goes back to the board, same filter.
  def back_path
    params[:return_to] == "board" ? household_root_path(show: params[:show].presence_in(%w[mine free])) : household_completions_path
  end
end
