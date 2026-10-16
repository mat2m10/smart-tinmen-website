require "test_helper"

class ChoreTest < ActiveSupport::TestCase
  test "a repeating chore restarts from the day it was done" do
    chore = chores(:toilet)
    chore.complete!(users(:matthieu), on: Date.new(2026, 10, 20))
    assert_equal Date.new(2026, 10, 27), chore.reload.due_on
    assert_nil chore.done_at
    assert_equal 1, chore.completions.count
  end

  test "a fixed schedule keeps its calendar day, even when done late" do
    chore = chores(:rent)
    chore.update!(due_on: Date.new(2026, 10, 25))
    chore.complete!(users(:livia), on: Date.new(2026, 11, 28))
    assert_equal Date.new(2026, 12, 25), chore.reload.due_on
  end

  test "a one-off chore is closed when done" do
    chore = chores(:gp)
    chore.complete!(users(:matthieu))
    assert chore.reload.done_at
    assert_not_includes Chore.pending, chore
  end

  test "completions keep the effort and planning flag" do
    chores(:rent).complete!(users(:livia))
    completion = Completion.last
    assert_equal 1, completion.effort
    assert completion.planning
  end

  test "buckets follow the due date" do
    today = Date.new(2026, 10, 16)
    chore = Chore.new(due_on: today - 1)
    assert_equal :overdue, chore.bucket(today)
    chore.due_on = today
    assert_equal :today, chore.bucket(today)
    chore.due_on = today + 6
    assert_equal :week, chore.bucket(today)
    chore.due_on = today + 7
    assert_equal :later, chore.bucket(today)
  end

  test "rejects unknown areas, rooms and an assignee that does not exist" do
    chore = Chore.new(title: "x", due_on: Date.current, area: "garage", room: "Attic", assignee_id: 0)
    assert_not chore.valid?
    assert chore.errors[:area].any?
    assert chore.errors[:room].any?
    assert chore.errors[:assignee].any?
  end

  test "repeat label" do
    assert_equal "Every week", chores(:toilet).repeat_label
    assert_equal "Once", chores(:gp).repeat_label
    assert_equal "Every 2 weeks", Chore.new(repeat_unit: "week", repeat_every: 2).repeat_label
  end
end
