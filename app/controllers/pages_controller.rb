class PagesController < ApplicationController
  def home
  end

  def sitemap
    expires_in 1.day, public: true
  end
end
