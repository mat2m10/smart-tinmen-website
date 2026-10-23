require "test_helper"

class PagesControllerTest < ActionDispatch::IntegrationTest
  test "should get home" do
    get root_url
    assert_response :success
  end

  test "home shows every section, the real projects and both founders" do
    get root_url
    %w[work services process team contact].each do |id|
      assert_select "section##{id}"
    end
    assert_select ".st-project", 6
    assert_select ".st-project h3", text: "Pangolin detection from the air"
    assert_select ".st-person h3", text: "Harold Loop"
    assert_select ".st-person h3", text: "Matthieu de Hemptinne"
    assert_select "a[href='mailto:info@smarttinmen.com']"
  end

  test "should get sitemap" do
    get sitemap_url
    assert_response :success
    assert_equal "application/xml", response.media_type
    assert_includes response.body, "<loc>#{root_url}</loc>"
  end

  test "link previews use the 1200x630 image" do
    get root_url
    assert_select "meta[property='og:image'][content*='og-image']"
    assert_select "meta[property='og:image:width'][content='1200']"
    assert_select "meta[name='twitter:card'][content='summary_large_image']"
  end
end
