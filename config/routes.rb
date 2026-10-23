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
        post :assign
      end
    end

    get "agenda" => "agenda#show", as: :agenda
    resource :profile, only: %i[show update]
    resources :completions, path: "done", only: %i[index update destroy]
    resources :notes, only: %i[index create destroy]
    resource :push_subscription, only: %i[create destroy]
    get "manifest.webmanifest" => "pwa#manifest", as: :manifest
    get "service-worker.js" => "pwa#service_worker", as: :service_worker
    get "calendar.ics" => "calendars#show", as: :calendar
    resources :attachments, only: :show

    namespace :api, defaults: { format: :json } do
      resources :chores, only: %i[index create update]
      get "completions" => "completions#index"
      get "events" => "events#index"
      resources :notes, only: %i[index create]
      resources :attachments, only: %i[show update]
      post "push" => "pushes#create"
    end
  end
end
