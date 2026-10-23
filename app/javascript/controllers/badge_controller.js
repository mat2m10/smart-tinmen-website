import { Controller } from "@hotwired/stimulus"

// Keeps the number on the app icon in step with the board: every page sets
// it to the chores waiting for you today, so ticking one off lowers it.
export default class extends Controller {
  static values = { count: Number }

  connect() {
    if (!navigator.setAppBadge) return
    const update = this.countValue > 0 ? navigator.setAppBadge(this.countValue) : navigator.clearAppBadge()
    update.catch(() => {})
  }
}
