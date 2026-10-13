import { Controller } from "@hotwired/stimulus"

// Copies the email address to the clipboard; falls back to selecting it.
export default class extends Controller {
  static targets = ["source", "button"]

  copy() {
    const text = this.sourceTarget.textContent.trim()
    navigator.clipboard.writeText(text)
      .then(() => { this.buttonTarget.textContent = "Copied" })
      .catch(() => this.select())
  }

  select() {
    const range = document.createRange()
    range.selectNodeContents(this.sourceTarget)
    const selection = window.getSelection()
    selection.removeAllRanges()
    selection.addRange(range)
    this.buttonTarget.textContent = "Selected, press Ctrl+C"
  }
}
