class CreateTabletopTables < ActiveRecord::Migration[8.1]
  def change
    # Game data (cards, points, army rules) is imported from a private YAML file
    # with bin/rails tabletop:import, so none of it lives in this repository.
    create_table :game_factions do |t|
      t.string :game, null: false
      t.string :name, null: false
      t.jsonb :rules, null: false, default: {}
      t.timestamps
    end
    add_index :game_factions, %i[game name], unique: true

    create_table :game_units do |t|
      t.string :game, null: false
      t.string :faction, null: false
      t.string :name, null: false
      t.string :subtitle
      t.string :category, null: false
      t.integer :points
      t.integer :points_printed
      t.integer :minis, null: false, default: 1
      t.boolean :unique, null: false, default: false
      t.jsonb :slots, null: false, default: []
      t.jsonb :keywords, null: false, default: []
      t.jsonb :stats, null: false, default: {}
      t.integer :selection_cost
      t.text :notes
      t.boolean :check, null: false, default: false
      t.timestamps
    end
    add_index :game_units, %i[game faction name], unique: true

    create_table :game_upgrades do |t|
      t.string :game, null: false
      t.string :faction, null: false
      t.string :name, null: false
      t.string :slot, null: false
      t.integer :points
      t.integer :points_printed
      t.string :restriction
      t.boolean :unique, null: false, default: false
      t.integer :limited
      t.boolean :adds_mini, null: false, default: false
      t.integer :minis_added
      t.boolean :removed, null: false, default: false
      t.text :notes
      t.boolean :check, null: false, default: false
      t.timestamps
    end
    add_index :game_upgrades, %i[game faction name], unique: true

    create_table :army_lists do |t|
      t.references :user, null: false, foreign_key: { on_delete: :cascade }
      t.string :game, null: false
      t.string :faction, null: false
      t.string :name, null: false
      t.integer :points_limit
      t.integer :ploys, null: false, default: 0
      t.text :notes
      t.timestamps
    end

    create_table :army_list_entries do |t|
      t.references :army_list, null: false, foreign_key: { on_delete: :cascade }
      t.references :game_unit, null: false, foreign_key: true
      t.integer :position, null: false, default: 0
      t.jsonb :upgrade_ids, null: false, default: []
      t.string :loadout
      t.timestamps
    end

    # What you own: a unit, or an upgrade that adds a miniature (like a squad leader).
    create_table :collection_items do |t|
      t.references :user, null: false, foreign_key: { on_delete: :cascade }
      t.references :item, polymorphic: true, null: false, index: false
      t.integer :built, null: false, default: 0
      t.integer :painted, null: false, default: 0
      t.integer :unbuilt, null: false, default: 0
      t.string :note
      t.timestamps
    end
    add_index :collection_items, %i[user_id item_type item_id], unique: true
  end
end
