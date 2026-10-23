import { Controller } from "@hotwired/stimulus"

// Swipe a row to the left to show its actions (Edit, Delete), like in a mail
// app. Swiping back, tapping the row or opening another row closes it again.
// touch-action: pan-y on the row leaves vertical scrolling to the browser.
export default class extends Controller {
  static targets = [ "row", "actions" ]

  connect() {
    this.offset = 0
    this.closeOthers = event => { if (event.detail !== this) this.close() }
    document.addEventListener("swipe:open", this.closeOthers)
  }

  disconnect() {
    document.removeEventListener("swipe:open", this.closeOthers)
  }

  start(event) {
    if (event.pointerType === "mouse" && event.button !== 0) return
    this.startX = event.clientX
    this.startY = event.clientY
    this.base = this.offset
    this.horizontal = null
  }

  move(event) {
    if (this.startX === undefined) return
    const dx = event.clientX - this.startX
    const dy = event.clientY - this.startY
    if (this.horizontal === null) {
      if (Math.abs(dx) < 8 && Math.abs(dy) < 8) return
      this.horizontal = Math.abs(dx) > Math.abs(dy)
      if (!this.horizontal) return this.reset()
      this.rowTarget.setPointerCapture(event.pointerId)
      this.element.classList.add("is-dragging")
    }
    this.offset = Math.min(0, Math.max(-this.width, this.base + dx))
    this.paint()
  }

  end() {
    if (this.horizontal) {
      this.element.classList.remove("is-dragging")
      this.swipedAt = Date.now()
      this.offset < -this.width / 3 ? this.open() : this.close()
    }
    this.reset()
  }

  // A swipe must not also follow the link it started on; a tap on an open row closes it.
  click(event) {
    if (Date.now() - (this.swipedAt || 0) < 400 || this.offset !== 0) {
      event.preventDefault()
      event.stopPropagation()
      if (Date.now() - (this.swipedAt || 0) >= 400) this.close()
    }
  }

  open() {
    document.dispatchEvent(new CustomEvent("swipe:open", { detail: this }))
    this.offset = -this.width
    this.paint()
  }

  close() {
    this.offset = 0
    this.paint()
  }

  reset() {
    this.startX = undefined
    this.horizontal = null
  }

  get width() {
    return this.actionsTarget.offsetWidth
  }

  paint() {
    this.rowTarget.style.transform = this.offset ? `translateX(${this.offset}px)` : ""
    this.element.classList.toggle("is-open", this.offset !== 0)
  }
}
