# One army or kill team and its building rules (rank limits, selection rules),
# imported with the cards from a private YAML file (bin/rails tabletop:import).
class GameFaction < ApplicationRecord
  GAMES = { "legion" => "Star Wars: Legion", "killteam" => "Kill Team" }.freeze

  validates :game, inclusion: { in: GAMES.keys }
  validates :name, presence: true, uniqueness: { scope: :game }

  def label
    "#{GAMES[game]} · #{name}"
  end
end
