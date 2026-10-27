class Household::ChoresController < Household::BaseController
  BUCKETS = { overdue: "Overdue", today: "Today", week: "This week", later: "Later" }.freeze

  before_action :set_chore, only: %i[edit update destroy complete claim assign]

  def index
    @filter = params[:show].presence_in(%w[mine free]) || "all"
    chores = Chore.visible_to(current_user).pending.includes(:assignee, :completions, :remarks).order(:due_on, :title)
    chores = chores.where(assignee: current_user) if @filter == "mine"
    chores = chores.where(assignee_id: nil) if @filter == "free"
    # Your personal chores get their own section at the top, whatever their date.
    @personal, shared = chores.reject(&:resting?).partition(&:personal)
    @buckets = shared.group_by(&:bucket)

    # The balance compares shared work only; personal chores count for no one.
    recent = Completion.where(completed_on: 30.days.ago.to_date.., chore: Chore.shared)
    @users = User.order(:name, :email)
    @effort = recent.group(:user_id).sum(:effort)
    @planning = recent.where(planning: true).group(:user_id).count
    @recently_done = Completion.where(chore: Chore.visible_to(current_user)).includes(:chore, :user).order(created_at: :desc).limit(8)
    @note = Note.new
  end

  def new
    @chore = Chore.new(due_on: Date.current, area: params[:area].presence || "Home & repairs")
  end

  def create
    @chore = Chore.new(chore_params.merge(created_by: current_user))
    @chore.assignee = current_user if @chore.personal
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
    @chore.assign_attributes(chore_params)
    if @chore.personal
      # Only your own or a free chore can become personal, or it would vanish for its owner.
      owner_id = @chore.assignee_id_in_database
      @chore.errors.add(:base, "Only your own chores or ones up for grabs can be personal") if owner_id && owner_id != current_user.id
      @chore.assignee = current_user
    end
    if @chore.errors.empty? && @chore.save
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
    # Its ticks go with it (dependent: :destroy), so its points leave the balance too.
    redirect_to household_root_path(show: filter_param), notice: "Deleted: #{@chore.title}", status: :see_other
  end

  # The button sends how many times the chore had been done when the page was
  # shown, so a double tap or a second phone with an old page counts once.
  # "by" ticks it off for the person it belongs to when they forgot to.
  def complete
    doer = current_user
    doer = @chore.assignee if params[:by].present? && params[:by].to_i == @chore.assignee_id
    done = @chore.with_lock do
      next false if @chore.done_at? || (params[:seen].present? && params[:seen].to_i != @chore.completions.count)

      completion = @chore.complete!(doer)
      Event.log!("chore_done", user: current_user, chore: @chore, summary: (doer == current_user ? nil : "#{@chore.title} (done by #{doer.display_name})"))
      completion
    end
    if done
      flash[:undo] = done.id
      sentence = @chore.done_sentence(doer.display_name)
      notify_others(doer == current_user ? sentence : "#{sentence} (ticked off by #{current_user.display_name})")
    end
    notice = "#{done ? "Done" : "Already done"}: #{@chore.title}"
    redirect_to household_root_path(show: filter_param), notice: notice, status: :see_other
  end

  # Take the chore, or let it go again if it is already yours. A personal
  # chore always stays with its owner.
  def claim
    return redirect_to(household_root_path(show: filter_param), status: :see_other) if @chore.personal

    @chore.update!(assignee: @chore.assignee == current_user ? nil : current_user)
    Event.log!(@chore.assignee ? "chore_claimed" : "chore_released", user: current_user, chore: @chore)
    redirect_to household_root_path(show: filter_param), status: :see_other
  end

  # Give the chore to someone (or back to "up for grabs") from the avatar menu.
  # The new owner hears about it when it is not you.
  def assign
    return redirect_to(household_root_path(show: filter_param), status: :see_other) if @chore.personal

    assignee = params[:assignee_id].present? ? User.find(params[:assignee_id]) : nil
    if assignee != @chore.assignee
      @chore.update!(assignee: assignee)
      Event.log!(assignee ? "chore_assigned" : "chore_released", user: current_user, chore: @chore,
                 summary: assignee && "#{@chore.title} (to #{assignee.display_name})")
      if assignee && assignee != current_user
        HouseholdPushJob.perform_later([ assignee.id ], "#{current_user.display_name} gave you “#{@chore.title}”", "", household_root_path)
      end
    end
    redirect_to household_root_path(show: filter_param), status: :see_other
  end

  private

  def set_chore
    @chore = Chore.visible_to(current_user).find(params[:id])
  end

  def load_notes
    @notes = @chore.remarks.includes(:user, :attachments).order(:created_at, :id)
    @note = Note.new(chore: @chore)
  end

  # One plain sentence, no body: "Matthieu emptied the dishwasher". Personal
  # chores stay quiet.
  def notify_others(sentence)
    ids = @chore.audience(except: current_user).ids
    HouseholdPushJob.perform_later(ids, sentence, "", household_root_path) if ids.any?
  end

  def filter_param
    params[:show].presence_in(%w[mine free])
  end

  def chore_params
    params.expect(chore: %i[title notes area room due_on due_time repeat_unit repeat_every fixed_schedule effort planning assignee_id personal])
  end
end
