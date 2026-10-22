import { Controller } from "@hotwired/stimulus"

// Shrinks photos on the phone before they are uploaded (longest side 1600 px,
// JPEG), so they stay small in the database, and shows thumbnails.
export default class extends Controller {
  static targets = [ "input", "previews" ]
  static values = { maxSide: { type: Number, default: 1600 }, max: { type: Number, default: 4 } }

  async pick() {
    const files = Array.from(this.inputTarget.files).slice(0, this.maxValue)
    const shrunk = new DataTransfer()
    this.previewsTarget.replaceChildren()
    for (const file of files) {
      const small = await this.shrink(file).catch(() => file)
      shrunk.items.add(small)
      const thumb = document.createElement("img")
      thumb.src = URL.createObjectURL(small)
      thumb.alt = ""
      this.previewsTarget.append(thumb)
    }
    this.inputTarget.files = shrunk.files
  }

  async shrink(file) {
    const bitmap = await createImageBitmap(file)
    const scale = Math.min(1, this.maxSideValue / Math.max(bitmap.width, bitmap.height))
    const canvas = document.createElement("canvas")
    canvas.width = Math.round(bitmap.width * scale)
    canvas.height = Math.round(bitmap.height * scale)
    canvas.getContext("2d").drawImage(bitmap, 0, 0, canvas.width, canvas.height)
    const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.82))
    if (!blob) return file
    return new File([ blob ], file.name.replace(/\.[^.]*$/, "") + ".jpg", { type: "image/jpeg" })
  }
}
