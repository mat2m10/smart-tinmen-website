Rails.application.routes.draw do
  get "up" => "rails/health#show", as: :rails_health_check
  root "pages#home"
  get "sitemap.xml" => "pages#sitemap", format: false, defaults: { format: "xml" }, as: :sitemap

  # Household app: private, not linked from the public site.
  get "household/login" => "household/sessions#new", as: :household_login
  get "auth/google_oauth2/callback" => "household/sessions#create"
  get "auth/failure" => "household/sessions#failure"
  delete "household/logout" => "household/sessions#destroy", as: :household_logout

  namespace :household do
    root "chores#index"
    resources :chores, except: %i[index show] do
      member do
        post :complete
        post :claim
      end
    end

    resources :notes, only: %i[index create destroy]
    resources :attachments, only: :show

    namespace :api, defaults: { format: :json } do
      resources :chores, only: %i[index create update]
      get "completions" => "completions#index"
      get "events" => "events#index"
      resources :notes, only: %i[index create]
      resources :attachments, only: %i[show update]
    end
  end
end
