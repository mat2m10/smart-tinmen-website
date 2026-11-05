namespace :household do
  desc "Create household users and tasks from YAML on stdin. Existing titles are left untouched. " \
       "Usage: bin/rails household:import < tasks.yml"
  task import: :environment do
    data = YAML.safe_load($stdin.read, permitted_classes: [ Date ]) || {}
    home = Home.for_founders

    Array(data["users"]).each do |row|
      user = User.find_or_initialize_by(email: row.fetch("email"))
      user.name ||= row["name"]
      user.home ||= home
      user.save!
    end

    created = 0
    Array(data["chores"]).each do |row|
      attrs = row.to_h
      next if home.chores.exists?(title: attrs["title"])

      email = attrs.delete("assignee_email")
      assignee = email.present? ? home.users.find_by!(email: email.strip.downcase) : nil
      home.chores.create!(attrs.merge("assignee" => assignee))
      created += 1
    end
    puts "Users: #{home.users.count}. Tasks created: #{created}, total open: #{home.chores.pending.count}."
  end
end
