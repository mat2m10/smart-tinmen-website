class Household::ChoresController < Household::BaseController
  BUCKETS = { overdue: "Overdue", today: "Today", week: "This week", later: "Later" }.freeze

  before_action :set_chore, only: %i[edit update destroy complete claim]

  def index
    @filter = params[:show].presence_in(%w[mine free]) || "all"
    chores = Chore.pending.includes(:assignee, :completions, :remarks).order(:due_on, :title)
    chores = chores.where(assignee: current_user) if @filter == "mine"
    chores = chores.where(assignee_id: nil) if @filter == "free"
    @buckets = chores.group_by(&:bucket)

    recent = Completion.where(completed_on: 30.days.ago.to_date..)
    @users = User.order(:name, :email)
    @effort = recent.group(:user_id).sum(:effort)
    @planning = recent.where(planning: true).group(:user_id).count
    @recently_done = Completion.includes(:chore, :user).order(created_at: :desc).limit(8)
    @latest_notes = Note.includes(:user, :chore, :attachments).recent_first.limit(3)
    @note = Note.new
  end

  def new
    @chore = Chore.new(due_on: Date.current, area: params[:area].presence || "Home & repairs")
  end

  def create
    @chore = Chore.new(chore_params.merge(created_by: current_user))
    if @chore.save
      Event.log!("chore_added", user: current_user, chore: @chore)
      notify_others("#{current_user.display_name} added “#{@chore.title}”")
      redirect_to household_root_path, notice: "Added: #{@chore.title}", status: :see_other
    else
      render :new, status: :unprocessable_entity
    end
  end

  def edit
    load_notes
  end

  def update
    if @chore.update(chore_params)
      Event.log!("chore_edited", user: current_user, chore: @chore)
      redirect_to household_root_path, notice: "Saved: #{@chore.title}", status: :see_other
    else
      load_notes
      render :edit, status: :unprocessable_entity
    end
  end

  def destroy
    @chore.destroy!
    Event.log!("chore_deleted", user: current_user, chore: @chore)
    redirect_to household_root_path, notice: "Deleted: #{@chore.title}", status: :see_other
  end

  # The button sends how many times the chore had been done when the page was
  # shown, so a double tap or a second phone with an old page counts once.
  def complete
    done = @chore.with_lock do
      next false if @chore.done_at? || (params[:seen].present? && params[:seen].to_i != @chore.completions.count)

      @chore.complete!(current_user)
      Event.log!("chore_done", user: current_user, chore: @chore)
      true
    end
    notify_others(@chore.done_sentence(current_user.display_name)) if done
    notice = "#{done ? "Done" : "Already done"}: #{@chore.title}"
    redirect_to household_root_path(show: filter_param), notice: notice, status: :see_other
  end

  # Take the chore, or let it go again if it is already yours.
  def claim
    @chore.update!(assignee: @chore.assignee == current_user ? nil : current_user)
    Event.log!(@chore.assignee ? "chore_claimed" : "chore_released", user: current_user, chore: @chore)
    redirect_to household_root_path(show: filter_param), status: :see_other
  end

  private

  def set_chore
    @chore = Chore.find(params[:id])
  end

  def load_notes
    @notes = @chore.remarks.includes(:user, :attachments).order(:created_at, :id)
    @note = Note.new(chore: @chore)
  end

  # One plain sentence, no body: "Matthieu emptied the dishwasher".
  def notify_others(sentence)
    HouseholdPushJob.perform_later(User.where.not(id: current_user.id).ids, sentence, "", household_root_path)
  end

  def filter_param
    params[:show].presence_in(%w[mine free])
  end

  def chore_params
    params.expect(chore: %i[title notes area room due_on due_time repeat_unit repeat_every fixed_schedule effort planning assignee_id])
  end
end
