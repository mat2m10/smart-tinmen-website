import { Controller } from "@hotwired/stimulus"

// Turns the chosen picture into a small square JPEG (centre crop) on the
// phone before it is uploaded, and shows it in place of the avatar.
// iPhone photos (HEIC) come out as JPEG too.
export default class extends Controller {
  static targets = [ "input", "preview" ]
  static values = { side: { type: Number, default: 256 } }

  async pick() {
    const file = this.inputTarget.files[0]
    if (!file) return
    const square = await this.crop(file).catch(() => null)
    if (square) {
      const files = new DataTransfer()
      files.items.add(square)
      this.inputTarget.files = files.files
    }
    const shown = square || file
    const img = document.createElement("img")
    img.src = URL.createObjectURL(shown)
    img.alt = ""
    this.previewTarget.replaceChildren(img)
    this.previewTarget.classList.add("hh-avatar--photo")
  }

  async crop(file) {
    const bitmap = await createImageBitmap(file)
    const size = Math.min(bitmap.width, bitmap.height)
    const canvas = document.createElement("canvas")
    canvas.width = canvas.height = Math.min(this.sideValue, size)
    canvas.getContext("2d").drawImage(bitmap, (bitmap.width - size) / 2, (bitmap.height - size) / 2, size, size, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.85))
    return blob && new File([ blob ], "avatar.jpg", { type: "image/jpeg" })
  }
}
