# The tabletop list builder (/tabletop): same Google sign-in, same members and
# the same privacy as the household. Every list and collection is per member.
class Tabletop::BaseController < Household::BaseController
  layout "tabletop"

  private

  def find_list(id)
    current_user.army_lists.find(id)
  end
end
