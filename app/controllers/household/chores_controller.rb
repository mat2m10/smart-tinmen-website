class Household::ChoresController < Household::BaseController
  BUCKETS = { overdue: "Overdue", today: "Today", week: "This week", later: "Later" }.freeze

  before_action :set_chore, only: %i[edit update destroy complete claim]

  def index
    @filter = params[:show].presence_in(%w[mine free]) || "all"
    chores = Chore.pending.includes(:assignee).order(:due_on, :title)
    chores = chores.where(assignee: current_user) if @filter == "mine"
    chores = chores.where(assignee_id: nil) if @filter == "free"
    @buckets = chores.group_by(&:bucket)

    recent = Completion.where(completed_on: 30.days.ago.to_date..)
    @users = User.order(:name, :email)
    @effort = recent.group(:user_id).sum(:effort)
    @planning = recent.where(planning: true).group(:user_id).count
    @recently_done = Completion.includes(:chore, :user).order(created_at: :desc).limit(8)
  end

  def new
    @chore = Chore.new(due_on: Date.current, area: params[:area].presence_in(Chore::AREAS.keys) || "home")
  end

  def create
    @chore = Chore.new(chore_params.merge(created_by: current_user))
    if @chore.save
      redirect_to household_root_path, notice: "Added: #{@chore.title}", status: :see_other
    else
      render :new, status: :unprocessable_entity
    end
  end

  def edit
  end

  def update
    if @chore.update(chore_params)
      redirect_to household_root_path, notice: "Saved: #{@chore.title}", status: :see_other
    else
      render :edit, status: :unprocessable_entity
    end
  end

  def destroy
    @chore.destroy!
    redirect_to household_root_path, notice: "Deleted: #{@chore.title}", status: :see_other
  end

  def complete
    @chore.complete!(current_user)
    redirect_to household_root_path(show: filter_param), notice: "Done: #{@chore.title}", status: :see_other
  end

  # Take the chore, or let it go again if it is already yours.
  def claim
    @chore.update!(assignee: @chore.assignee == current_user ? nil : current_user)
    redirect_to household_root_path(show: filter_param), status: :see_other
  end

  private

  def set_chore
    @chore = Chore.find(params[:id])
  end

  def filter_param
    params[:show].presence_in(%w[mine free])
  end

  def chore_params
    params.expect(chore: %i[title notes area room due_on repeat_unit repeat_every fixed_schedule effort planning assignee_id])
  end
end
