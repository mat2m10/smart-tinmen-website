namespace :tabletop do
  desc "Import or update one faction's cards and army rules from YAML on stdin. " \
       "Usage: bin/rails tabletop:import < faction.yml"
  task import: :environment do
    puts Tabletop::Importer.new($stdin.read).run
  end

  desc "Set a member's collection counts from YAML on stdin (game + items). " \
       "Usage: EMAIL=you@example.com bin/rails tabletop:collection < collection.yml"
  task :collection, [ :email ] => :environment do |_, args|
    email = args[:email].presence || ENV["EMAIL"].presence || abort("Give the member's email: EMAIL=you@example.com")
    user = User.find_by(email: email.strip.downcase) || abort("No user #{email}: sign in once at /household first.")
    result = Tabletop::CollectionImporter.new(user, $stdin.read).run
    puts "Collection of #{user.email}: #{result.updated} items set."
    puts "Unknown names (not imported): #{result.unknown.join(", ")}" if result.unknown.any?
  end
end
