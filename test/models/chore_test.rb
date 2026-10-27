require "test_helper"

class ChoreTest < ActiveSupport::TestCase
  include ActiveJob::TestHelper

  test "a repeating chore restarts from the day it was done" do
    chore = chores(:toilet)
    chore.complete!(users(:alice), on: Date.new(2026, 10, 20))
    assert_equal Date.new(2026, 10, 27), chore.reload.due_on
    assert_nil chore.done_at
    assert_equal 1, chore.completions.count
  end

  test "a weekday chore skips the weekend" do
    chore = Chore.create!(title: "Take the kid to daycare", area: "Kids", due_on: Date.new(2026, 10, 24), repeat_unit: "weekday", effort: 1)
    assert_equal Date.new(2026, 10, 26), chore.due_on # created on a Saturday: starts on Monday
    chore.update!(due_on: Date.new(2026, 10, 23))
    chore.complete!(users(:alice), on: Date.new(2026, 10, 23)) # a Friday
    assert_equal Date.new(2026, 10, 26), chore.reload.due_on # Monday
    chore.complete!(users(:bob), on: Date.new(2026, 10, 26))
    assert_equal Date.new(2026, 10, 27), chore.reload.due_on
    chore.complete!(users(:bob), on: Date.new(2026, 10, 31)) # done late, on a Saturday
    assert_equal Date.new(2026, 11, 2), chore.reload.due_on
    assert_equal "Every weekday", chore.repeat_label
    assert_equal [ Date.new(2026, 11, 2), Date.new(2026, 11, 3) ], chore.occurrences(Date.new(2026, 11, 1)..Date.new(2026, 11, 3))
  end

  test "a fixed schedule keeps its calendar day, even when done late" do
    chore = chores(:rent)
    chore.update!(due_on: Date.new(2026, 10, 25))
    chore.complete!(users(:bob), on: Date.new(2026, 11, 28))
    assert_equal Date.new(2026, 12, 25), chore.reload.due_on
  end

  test "a one-off chore is closed when done" do
    chore = chores(:gp)
    chore.complete!(users(:alice))
    assert chore.reload.done_at
    assert_not_includes Chore.pending, chore
  end

  test "completions keep the effort and planning flag" do
    chores(:rent).complete!(users(:bob))
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

  test "needs an area, keeps room names short, rejects an assignee that does not exist" do
    chore = Chore.new(title: "x", due_on: Date.current, area: "", room: "x" * 41, assignee_id: 0)
    assert_not chore.valid?
    assert chore.errors[:area].any?
    assert chore.errors[:room].any?
    assert chore.errors[:assignee].any?
  end

  test "area suggestions mix the defaults with what the household already uses" do
    Chore.create!(title: "Wash the car", area: "Car", due_on: Date.current)
    assert_includes Chore.area_options, "Cleaning"
    assert_includes Chore.area_options, "Car"
    assert_equal Chore.area_options.uniq, Chore.area_options
    assert_includes Chore.room_options, "Bathroom"
  end

  test "repeat label" do
    assert_equal "Every week", chores(:toilet).repeat_label
    assert_equal "Once", chores(:gp).repeat_label
    assert_equal "Every 2 weeks", Chore.new(repeat_unit: "week", repeat_every: 2).repeat_label
  end

  test "occurrences list the due date and the coming repeats in range" do
    chore = chores(:toilet)
    chore.update!(due_on: Date.new(2026, 10, 20))
    assert_equal [ Date.new(2026, 10, 20), Date.new(2026, 10, 27) ], chore.occurrences(Date.new(2026, 10, 19)..Date.new(2026, 11, 2))
    assert_equal [ Date.new(2026, 10, 27) ], chore.occurrences(Date.new(2026, 10, 21)..Date.new(2026, 10, 30))
    assert_equal [ Date.new(2026, 10, 20) ], chore.occurrences(Date.new(2026, 10, 20)...Date.new(2026, 10, 27))
    assert_equal [], chores(:gp).occurrences((chores(:gp).due_on + 1)..(chores(:gp).due_on + 30))
  end

  test "due times are stored as HH:MM and checked" do
    chore = chores(:gp)
    chore.update!(due_time: " 9:30 ")
    assert_equal "09:30", chore.due_time
    chore.update!(due_time: "14:05:00")
    assert_equal "14:05", chore.due_time
    chore.update!(due_time: "")
    assert_nil chore.due_time
    assert_not chore.update(due_time: "25:00")
    assert_not chore.update(due_time: "noon")
    assert_not chore.update(due_time: "7:05pm")
  end

  test "done sentences read naturally, and quote titles that don't start with a known verb" do
    sentence = ->(title) { Chore.new(title: title).done_sentence("Matthieu") }
    assert_equal "Matthieu emptied the dishwasher", sentence.("Empty the dishwasher")
    assert_equal "Matthieu took out the bins", sentence.("Take out the bins")
    assert_equal "Matthieu tidied the hall", sentence.("tidy the hall")
    assert_equal "Matthieu prepared lunch boxes", sentence.("Prepare lunch boxes")
    assert_equal "Matthieu paid", sentence.("Pay")
    assert_equal "Matthieu finished “Groceries”", sentence.("Groceries")
  end

  test "timed chores sync to the calendar when their details change" do
    chore = chores(:gp)
    assert_no_enqueued_jobs(only: HouseholdCalendarJob) { chore.update!(title: "Find a doctor") }
    assert_enqueued_with(job: HouseholdCalendarJob, args: [ chore.id ]) { chore.update!(due_time: "09:00") }
    assert_enqueued_with(job: HouseholdCalendarJob, args: [ chore.id ]) { chore.update!(due_on: chore.due_on + 1) }
    assert_enqueued_with(job: HouseholdCalendarJob, args: [ chore.id ]) { chore.update!(due_time: nil) }
  end

  test "finishing a one-off appointment keeps its calendar event" do
    chore = chores(:gp)
    chore.update!(due_time: "09:00")
    assert_no_enqueued_jobs(only: HouseholdCalendarJob) { chore.complete!(users(:alice)) }
  end

  test "deleting a timed chore removes its calendar event" do
    chore = chores(:gp)
    chore.update!(due_time: "09:00")
    assert_enqueued_with(job: HouseholdCalendarJob, args: [ chore.id ]) { chore.destroy! }
  end

  test "a personal chore needs an owner and is hidden from the other member" do
    chore = chores(:gp)
    assert_not chore.update(personal: true, assignee: nil)
    chore.update!(personal: true, assignee: users(:bob))
    assert_includes Chore.visible_to(users(:bob)), chore
    assert_not_includes Chore.visible_to(users(:alice)), chore
    assert_includes Chore.visible_to(users(:alice)), chores(:toilet)
    assert_empty chore.audience(except: users(:bob))
  end

  test "complete! returns the new tick and undo! says whether the chore came back" do
    chore = chores(:toilet)
    older = chore.complete!(users(:alice))
    latest = chore.complete!(users(:bob))
    assert_equal [ users(:alice), users(:bob) ], [ older.user, latest.user ]
    assert_not chore.undo!(older)
    assert chore.undo!(latest)
  end

  test "editing a resting chore brings it back on the board" do
    chore = chores(:toilet)
    chore.complete!(users(:alice))
    assert chore.reload.resting?
    travel 2.seconds do
      chore.update!(due_on: Date.current + 2)
      assert_not chore.reload.resting?
    end
  end
end
