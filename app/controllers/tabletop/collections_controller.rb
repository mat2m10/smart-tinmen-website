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

  # The steppers save in the background and get JSON back, so the page is not
  # reloaded under your fingers; without JavaScript the form saves as a page.
  def update
    item = (params[:item_type] == "GameUpgrade" ? GameUpgrade : GameUnit).find(params[:item_id])
    record = current_user.collection_items.find_or_initialize_by(item: item)
    saved = record.update(params.expect(collection_item: CollectionItem::COUNTS))
    respond_to do |format|
      format.json do
        if saved
          render json: record.slice(*CollectionItem::COUNTS).merge(owned: record.owned)
        else
          render json: { errors: record.errors.full_messages }, status: :unprocessable_entity
        end
      end
      format.html do
        redirect_to tabletop_collection_path(game: item.game), alert: (record.errors.full_messages.to_sentence unless saved), status: :see_other
      end
    end
  end
end
