import { Controller } from "@hotwired/stimulus"

// Opens the browser's print dialog for the list card.
export default class extends Controller {
  print() {
    window.print()
  }
}
