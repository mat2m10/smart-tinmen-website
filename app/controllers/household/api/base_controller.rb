# Token-protected JSON API so an assistant can read the household list and add
# what was forgotten. The token lives in HOUSEHOLD_API_TOKEN (at least 32 characters)
# and reaches only the home that has the assistant; other homes stay out of reach.
class Household::Api::BaseController < ActionController::API
  include ActionController::HttpAuthentication::Token::ControllerMethods

  before_action :authenticate
  before_action :set_home

  private

  def set_home
    @home = Home.assistant_home
    head :not_found unless @home
  end

  def authenticate
    expected = ENV["HOUSEHOLD_API_TOKEN"].to_s
    authenticate_or_request_with_http_token do |token, _options|
      expected.length >= 32 && ActiveSupport::SecurityUtils.secure_compare(token, expected)
    end
  end
end
