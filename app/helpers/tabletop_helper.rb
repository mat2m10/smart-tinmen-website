module TabletopHelper
  CATEGORY_LABELS = { "special_forces" => "Special Forces", "extra" => "Extra selections", "broodcoven" => "Broodcoven" }.freeze

  def category_label(category)
    CATEGORY_LABELS[category] || category.to_s.humanize
  end

  def slot_label(slot)
    slot.to_s.humanize
  end

  def points_label(points)
    points.nil? ? "? pts" : "#{points} pts"
  end

  # Legion counts points, Kill Team counts operatives.
  def list_size(list)
    if list.legion?
      [ list.total_points, list.points_limit ].compact.join(" / ") + " pts"
    else
      pluralize(list.entries.size, "operative")
    end
  end

  def validity_badge(validator)
    if validator.valid?
      tag.span(class: "tt-badge tt-badge--ok") { safe_join([ tag.span("check", class: "ms", aria: { hidden: true }), "Valid" ]) }
    else
      tag.span(class: "tt-badge tt-badge--bad") { safe_join([ tag.span("error", class: "ms", aria: { hidden: true }), pluralize(validator.errors.size, "problem") ]) }
    end
  end

  # "Own 2 · 1 painted", marked when the list needs more than you have.
  def ownership(collection, card, needed)
    owned = collection.owned(card)
    painted = collection.painted(card)
    text = owned.zero? ? "Not owned" : [ "Own #{owned}", ("#{painted} painted" if painted.positive?) ].compact.join(" · ")
    tag.span(text, class: [ "tt-own", ("tt-own--short" if owned < needed) ], title: ("The list needs #{needed}" if owned < needed))
  end
end
