import { Controller } from "@hotwired/stimulus"

// Tappable suggestions for a text field: tapping a chip fills the field, and
// the chip that matches what is typed stays highlighted. Typing still works.
export default class extends Controller {
  static targets = ["input", "chip"]

  connect() {
    this.sync()
  }

  pick(event) {
    this.inputTarget.value = event.params.value
    this.sync()
  }

  sync() {
    const value = this.inputTarget.value.trim().toLowerCase()
    this.chipTargets.forEach((chip) => {
      chip.setAttribute("aria-pressed", String(chip.dataset.choiceValueParam.toLowerCase() === value))
    })
  }
}
