namespace :household do
  desc "Create household users and tasks from YAML on stdin. Existing titles are left untouched. " \
       "Usage: bin/rails household:import < tasks.yml"
  task import: :environment do
    data = YAML.safe_load($stdin.read, permitted_classes: [ Date ]) || {}

    Array(data["users"]).each do |row|
      user = User.find_or_initialize_by(email: row.fetch("email"))
      user.name ||= row["name"]
      user.save!
    end

    created = 0
    Array(data["chores"]).each do |row|
      attrs = row.to_h
      next if Chore.exists?(title: attrs["title"])

      email = attrs.delete("assignee_email")
      assignee = email.present? ? User.find_by!(email: email.strip.downcase) : nil
      Chore.create!(attrs.merge("assignee" => assignee))
      created += 1
    end
    puts "Users: #{User.count}. Tasks created: #{created}, total open: #{Chore.pending.count}."
  end
end
