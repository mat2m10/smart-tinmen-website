import { Controller } from "@hotwired/stimulus"

// Minus and plus buttons around the number fields of one collection row. The
// row saves itself in the background a moment after the last change (a few
// quick taps make one request), without reloading the page, so the next field
// keeps what you type and your focus. A pending save is sent right away when
// the row leaves the page.
export default class extends Controller {
  static targets = [ "owned", "status" ]

  connect() {
    this.sent = 0
    this.flush = this.flush.bind(this)
    window.addEventListener("pagehide", this.flush)
  }

  disconnect() {
    window.removeEventListener("pagehide", this.flush)
    this.flush()
  }

  step(event) {
    const input = event.currentTarget.closest("[data-stepper-field]").querySelector("input")
    input.value = Math.max(0, (parseInt(input.value, 10) || 0) + event.params.delta)
    this.save()
  }

  save() {
    this.pending = true
    clearTimeout(this.timer)
    this.timer = setTimeout(this.flush, 600)
  }

  // Enter in a field saves now instead of submitting the form as a page visit.
  submit(event) {
    event.preventDefault()
    this.pending = true
    this.flush()
  }

  flush() {
    clearTimeout(this.timer)
    if (!this.pending) return
    this.pending = false
    this.send()
  }

  async send() {
    const request = ++this.sent
    try {
      const response = await fetch(this.element.action, {
        method: "POST", body: new FormData(this.element), keepalive: true,
        headers: { Accept: "application/json" }, credentials: "same-origin"
      })
      const data = await response.json()
      if (request !== this.sent) return
      if (response.ok) {
        this.ownedTarget.textContent = data.owned
        this.statusTarget.textContent = ""
      } else {
        this.statusTarget.textContent = data.errors.join(", ")
      }
    } catch {
      if (request === this.sent) this.statusTarget.textContent = "Not saved: check your connection."
    }
  }
}
