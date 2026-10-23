# Your lists, the list builder and the printable list card.
class Tabletop::ListsController < Tabletop::BaseController
  before_action :set_list, only: %i[show edit update destroy print]

  def index
    @lists = current_user.army_lists.includes(entries: :game_unit).order(updated_at: :desc).group_by(&:game)
  end

  def new
    @list = ArmyList.new
    @factions = GameFaction.order(:game, :name)
  end

  def create
    faction = GameFaction.find_by(id: params.dig(:army_list, :faction_id))
    @list = current_user.army_lists.new(name: params.dig(:army_list, :name), game: faction&.game, faction: faction&.name)
    if @list.save
      redirect_to tabletop_list_path(@list), status: :see_other
    else
      @factions = GameFaction.order(:game, :name)
      render :new, status: :unprocessable_entity
    end
  end

  def show
    @owned_only = params[:owned] == "1"
    @collection = Tabletop::Collection.new(current_user, @list.game)
    @units = @list.units.sorted.to_a
    @units.select! { |unit| @collection.owned(unit).positive? } if @owned_only
  end

  def edit
  end

  def update
    if @list.update(params.expect(army_list: %i[name points_limit notes ploys]))
      redirect_to tabletop_list_path(@list), status: :see_other
    else
      render :edit, status: :unprocessable_entity
    end
  end

  def destroy
    @list.destroy!
    redirect_to tabletop_root_path, notice: "List deleted", status: :see_other
  end

  def print
    @collection = Tabletop::Collection.new(current_user, @list.game)
  end

  private

  def set_list
    @list = find_list(params[:id])
  end
end
