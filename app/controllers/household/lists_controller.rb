# The Lists tab: shared lists of the home (groceries, gift ideas, a wish list).
class Household::ListsController < Household::BaseController
  def index
    @lists = List.visible_to(current_user).includes(:items).order(:name)
    @suggested = List::SUGGESTED.reject { |name| @lists.any? { |list| list.name.casecmp?(name) } }
  end

  def show
    @list = List.visible_to(current_user).find(params[:id])
    @items = @list.items.includes(:added_by).partition { |item| !item.ticked? }
  end

  def create
    list = current_user.home.lists.new(name: params.dig(:list, :name))
    if list.save
      redirect_to household_list_path(list), status: :see_other
    else
      redirect_to household_lists_path, alert: list.errors.full_messages.to_sentence, status: :see_other
    end
  end

  def destroy
    list = List.visible_to(current_user).find(params[:id])
    list.destroy!
    redirect_to household_lists_path, notice: "Deleted the list #{list.name}", status: :see_other
  end
end
