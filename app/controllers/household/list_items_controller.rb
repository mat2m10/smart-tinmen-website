# Adding, ticking and removing things on one of the home's lists.
class Household::ListItemsController < Household::BaseController
  before_action :set_list

  # One item, or one per line when a few are pasted at once.
  def create
    bodies = params.dig(:list_item, :body).to_s.split(/\r?\n/).map(&:squish).compact_blank.first(50)
    items = bodies.map { |body| @list.items.new(body: body, added_by: current_user) }
    if items.any? && items.all?(&:valid?)
      ListItem.transaction { items.each(&:save!) }
      # Keeps the keyboard up to add the next thing; opening or ticking does not.
      redirect_to household_list_path(@list, adding: 1), status: :see_other
    else
      problem = items.flat_map { |item| item.errors.full_messages }.first || "Type something to add"
      redirect_to household_list_path(@list), alert: problem, status: :see_other
    end
  end

  # Tick an item off, or bring it back.
  def update
    item = @list.items.find(params[:id])
    item.update!(checked_at: item.ticked? ? nil : Time.current)
    redirect_to household_list_path(@list), status: :see_other
  end

  def destroy
    @list.items.find(params[:id]).destroy!
    redirect_to household_list_path(@list), status: :see_other
  end

  # Remove everything already ticked off.
  def clear
    count = @list.items.ticked.delete_all
    redirect_to household_list_path(@list), notice: "Cleared #{count} ticked #{"item".pluralize(count)}", status: :see_other
  end

  private

  def set_list
    @list = List.visible_to(current_user).find(params[:list_id])
  end
end
