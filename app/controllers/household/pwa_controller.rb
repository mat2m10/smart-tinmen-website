# The web-app manifest and service worker that let the household app be
# installed on a phone and receive notifications. Neither contains anything
# private, so they are served without a login (browsers fetch them without cookies).
class Household::PwaController < ActionController::Base
  # The service worker is JavaScript fetched by the browser itself; Rails' guard
  # against cross-origin <script> embedding would refuse it. It holds no data.
  skip_forgery_protection

  def manifest
    response.headers["X-Robots-Tag"] = "noindex, nofollow"
    render json: {
      name: "Household", short_name: "Household", start_url: "/household", scope: "/household/", display: "standalone",
      background_color: "#F8FAFD", theme_color: "#F8FAFD",
      icons: [
        { src: helpers.asset_path("android-192.png"), sizes: "192x192", type: "image/png" },
        { src: helpers.asset_path("android-512.png"), sizes: "512x512", type: "image/png" }
      ]
    }, content_type: "application/manifest+json"
  end

  def service_worker
    response.headers["Service-Worker-Allowed"] = "/household/"
    render template: "household/pwa/service_worker", layout: false, content_type: "text/javascript", formats: [ :js ]
  end
end
