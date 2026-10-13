require "test_helper"

class PagesControllerTest < ActionDispatch::IntegrationTest
  test "should get home" do
    get root_url
    assert_response :success
  end

  test "home shows every section, the real projects and both founders" do
    get root_url
    %w[work services process record team contact].each do |id|
      assert_select "section##{id}"
    end
    assert_select ".st-project", 6
    assert_select ".st-project h3", text: "Pangolin detection from the air"
    assert_select ".st-person h3", text: "Harold Loop"
    assert_select ".st-person h3", text: "Matthieu de Hemptinne"
    assert_select "a[href='mailto:info@smarttinmen.com']"
  end
end
