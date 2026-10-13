Rails.application.routes.draw do
  get "up" => "rails/health#show", as: :rails_health_check
  root "pages#home"
  get "sitemap.xml" => "pages#sitemap", format: false, defaults: { format: "xml" }, as: :sitemap
end
