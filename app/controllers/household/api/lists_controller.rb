# Lets Claude read the home's lists (groceries, gift ideas...) and add to them,
# for example the shopping mentioned in a voice note. Claude never ticks or
# removes items: that stays with the people doing the shopping.
class Household::Api::ListsController < Household::Api::BaseController
  def index
    lists = @home.lists.includes(items: :added_by).order(:name)
    render json: lists.map { |list| list_json(list) }
  end

  # Adds items to a list, creating the list when it does not exist yet.
  # {"list":{"name":"Groceries","items":["Onions","Garlic"]}}
  def create
    attrs = params.expect(list: [ :name, items: [] ])
    name = attrs[:name].to_s.squish
    bodies = Array(attrs[:items]).map { |body| body.to_s.squish }.compact_blank.first(50)
    return render json: { errors: [ "items is required" ] }, status: :unprocessable_entity if bodies.empty?

    list = @home.lists.where("LOWER(name) = ?", name.downcase).first || @home.lists.new(name: name)
    items = bodies.map { |body| list.items.new(body: body) }
    if list.valid? && items.all?(&:valid?)
      ListItem.transaction { list.save! && items.each(&:save!) }
      render json: list_json(list.reload), status: :created
    else
      render json: { errors: (list.errors.full_messages + items.flat_map { |item| item.errors.full_messages }).uniq }, status: :unprocessable_entity
    end
  end

  private

  def list_json(list)
    { id: list.id, name: list.name,
      items: list.items.map { |item| { id: item.id, body: item.body, ticked: item.ticked?, added_by: item.added_by&.email, added_at: item.created_at } } }
  end
end
