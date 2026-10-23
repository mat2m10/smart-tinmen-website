module Tabletop
  # Sets one member's collection from YAML:
  #   game: legion
  #   faction: Some Faction   (optional, when a name exists in two factions)
  #   items:
  #     - {name: "Some Unit", painted: 4, built: 1, note: "optional"}
  # Names match a unit first, then an upgrade (for upgrade miniatures).
  # Counts left out are set to 0. Unknown names are reported, not created.
  class CollectionImporter
    Result = Struct.new(:updated, :unknown)

    def initialize(user, yaml)
      @user = user
      @data = YAML.safe_load(yaml) || {}
      @game = @data["game"].to_s
      @faction = @data["faction"].presence
      raise ArgumentError, "game must be one of #{GameFaction::GAMES.keys.join(", ")}" unless GameFaction::GAMES.key?(@game)
    end

    def run
      result = Result.new(0, [])
      ActiveRecord::Base.transaction do
        Array(@data["items"]).each do |row|
          item = find_item(row["name"].to_s.strip)
          next result.unknown << row["name"].to_s unless item

          record = @user.collection_items.find_or_initialize_by(item: item)
          record.update!(built: row["built"].to_i, painted: row["painted"].to_i, unbuilt: row["unbuilt"].to_i, note: row["note"])
          result.updated += 1
        end
      end
      result
    end

    private

    def find_item(name)
      [ GameUnit, GameUpgrade ].each do |model|
        scope = model.where(game: @game, name: name)
        scope = scope.where(faction: @faction) if @faction
        return scope.first if scope.exists?
      end
      nil
    end
  end
end
