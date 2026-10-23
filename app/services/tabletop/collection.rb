module Tabletop
  # One member's collection for one game, looked up per unit or upgrade.
  class Collection
    def initialize(user, game)
      units = user.collection_items.where(item_type: "GameUnit", item_id: GameUnit.where(game: game).select(:id))
      upgrades = user.collection_items.where(item_type: "GameUpgrade", item_id: GameUpgrade.where(game: game).select(:id))
      @items = units.or(upgrades).index_by { |item| [ item.item_type, item.item_id ] }
    end

    def item(card)
      @items[[ card.class.name, card.id ]]
    end

    def owned(card)
      item(card)&.owned.to_i
    end

    def painted(card)
      item(card)&.painted.to_i
    end
  end
end
