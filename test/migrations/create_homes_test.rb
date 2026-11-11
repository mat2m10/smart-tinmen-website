require "test_helper"
require Rails.root.join("db/migrate/20261105120000_create_homes").to_s
require Rails.root.join("db/migrate/20261111120000_create_lists").to_s

# The migration moves a live household's data, so it runs here on tables that
# hold rows. Everything, the schema changes too, is rolled back after the test.
class CreateHomesTest < ActiveSupport::TestCase
  MODELS = [ User, Chore, Note, Event, Home, List, ListItem ].freeze

  def after_teardown
    super
    reset_schema_caches
  end

  test "everything that exists moves into one home with the assistant" do
    note = Note.create!(body: "keep me", user: users(:alice))
    Event.log!("note_added", user: users(:alice), note: note)
    counts = %w[users chores notes events].index_with { |table| count("SELECT COUNT(*) FROM #{table}") }

    migrate(:down)
    assert_not connection.table_exists?(:homes)
    assert_not connection.column_exists?(:chores, :home_id)

    migrate(:up)
    assert_equal 1, count("SELECT COUNT(*) FROM homes")
    assert_equal 1, count("SELECT COUNT(*) FROM homes WHERE assistant")
    home_id = count("SELECT id FROM homes")
    counts.each do |table, rows|
      assert_equal rows, count("SELECT COUNT(*) FROM #{table}"), table
      assert_equal rows, count("SELECT COUNT(*) FROM #{table} WHERE home_id = #{home_id}"), table
    end
    assert_equal false, connection.columns(:chores).find { |column| column.name == "home_id" }.null
    assert_equal true, connection.columns(:users).find { |column| column.name == "home_id" }.null
  end

  private

  def connection
    ActiveRecord::Base.connection
  end

  def count(sql)
    connection.select_value(sql).to_i
  end

  # Lists point at homes, so they come off first and go back last.
  def migrate(direction)
    migrations = [ CreateHomes, CreateLists ]
    migrations.reverse! if direction == :down
    ActiveRecord::Migration.suppress_messages { migrations.each { |migration| migration.new.migrate(direction) } }
    reset_schema_caches
  end

  # Dropped and re-added columns make cached statements and column lists stale.
  def reset_schema_caches
    connection.clear_cache!
    connection.schema_cache.clear!
    MODELS.each(&:reset_column_information)
  end
end
