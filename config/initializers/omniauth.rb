# Google sign-in for the household app. Credentials come from Heroku config vars.
Rails.application.config.middleware.use OmniAuth::Builder do
  provider :google_oauth2, ENV["GOOGLE_CLIENT_ID"], ENV["GOOGLE_CLIENT_SECRET"],
    scope: "email,profile", prompt: "select_account"
end

OmniAuth.config.allowed_request_methods = [ :post ]
OmniAuth.config.on_failure = proc { |env| OmniAuth::FailureEndpoint.new(env).redirect_to_failure }
