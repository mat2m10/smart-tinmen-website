import { Controller } from "@hotwired/stimulus"

// Minus and plus buttons around number fields. The form saves itself a moment
// after the last tap, so a few quick taps make one request.
export default class extends Controller {
  step(event) {
    const input = event.currentTarget.closest("[data-stepper-field]").querySelector("input")
    input.value = Math.max(0, (parseInt(input.value, 10) || 0) + event.params.delta)
    this.save()
  }

  save() {
    clearTimeout(this.timer)
    this.timer = setTimeout(() => this.element.requestSubmit(), 600)
  }

  disconnect() {
    clearTimeout(this.timer)
  }
}
