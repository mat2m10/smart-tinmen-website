# What you own per game: built, painted and still in the box.
class Tabletop::CollectionsController < Tabletop::BaseController
  def show
    @game = GameFaction::GAMES.key?(params[:game]) ? params[:game] : (GameFaction.order(:game).pick(:game) || "legion")
    @factions = GameUnit.where(game: @game).order(:faction, :name).group_by(&:faction).transform_values do |units|
      units.group_by(&:category).sort_by { |category, _| GameUnit.category_rank(category) }
    end
    @upgrades = GameUpgrade.where(game: @game, adds_mini: true).order(:name).group_by(&:faction)
    @collection = Tabletop::Collection.new(current_user, @game)
  end

  def update
    item = (params[:item_type] == "GameUpgrade" ? GameUpgrade : GameUnit).find(params[:item_id])
    record = current_user.collection_items.find_or_initialize_by(item: item)
    if record.update(params.expect(collection_item: CollectionItem::COUNTS))
      redirect_to tabletop_collection_path(game: item.game), status: :see_other
    else
      redirect_to tabletop_collection_path(game: item.game), alert: record.errors.full_messages.to_sentence, status: :see_other
    end
  end
end
