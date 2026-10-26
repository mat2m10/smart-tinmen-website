# The Done page: what was ticked off in the last three weeks, so a wrong or
# test tick can be taken back or put on the right person.
class Household::CompletionsController < Household::BaseController
  before_action :set_completion, only: %i[update destroy]

  def index
    @completions = Completion.recent.where(chore: Chore.visible_to(current_user))
                             .includes(:user, chore: :assignee).order(completed_on: :desc, created_at: :desc)
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
    chore = @completion.chore
    chore.undo!(@completion)
    Event.log!("chore_undone", user: current_user, chore: chore)
    # Undo on the board's "Done" message goes back to the board.
    back = params[:return_to] == "board" ? household_root_path : household_completions_path
    redirect_to back, notice: "Not done: #{chore.title}. It is back on the list", status: :see_other
  end

  private

  def set_completion
    @completion = Completion.where(chore: Chore.visible_to(current_user)).find(params[:id])
  end
end
