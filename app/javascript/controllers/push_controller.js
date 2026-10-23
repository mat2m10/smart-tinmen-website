import { Controller } from "@hotwired/stimulus"

// Turns household notifications on or off for this phone. On iPhone this
// only works once the app is added to the home screen (iOS 16.4 and later).
// With invite (the board) the card only shows on phones that can still turn
// notifications on, and goes away once they are on; the profile page has the switch.
export default class extends Controller {
  static targets = [ "message", "button" ]
  static values = { key: String, invite: Boolean }

  async connect() {
    if (!this.keyValue) return
    const standalone = window.matchMedia("(display-mode: standalone)").matches || navigator.standalone
    if (!("serviceWorker" in navigator) || !("PushManager" in window)) {
      if (/iPhone|iPad/.test(navigator.userAgent) && !standalone && !this.inviteValue) {
        this.show("To get notifications on iPhone: tap Share, then “Add to Home Screen”, and open Household from there.", false)
      }
      return
    }
    let subscription
    try {
      this.registration = await navigator.serviceWorker.register("/household/service-worker.js", { scope: "/household" })
      subscription = await this.registration.pushManager.getSubscription()
      if (subscription) await this.save(subscription)
    } catch {
      if (!this.inviteValue) this.show("Notifications are not available right now.", false)
      return
    }
    if (this.inviteValue && (subscription || Notification.permission === "denied")) return
    if (subscription) {
      this.show("Notifications are on for this phone.", true, "Turn off")
    } else if (Notification.permission === "denied") {
      this.show("Notifications are blocked for this site in the browser settings.", false)
    } else {
      this.show("Get reminders and new notes on this phone.", true, "Turn on notifications")
    }
  }

  async toggle() {
    this.buttonTarget.disabled = true
    try {
      const current = await this.registration.pushManager.getSubscription()
      if (current) {
        await fetch("/household/push_subscription", { method: "DELETE", headers: this.headers(), body: JSON.stringify({ endpoint: current.endpoint }) })
        await current.unsubscribe()
        this.show("Notifications are off for this phone.", true, "Turn on notifications")
      } else {
        const subscription = await this.registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: this.decode(this.keyValue) })
        await this.save(subscription)
        this.show("Notifications are on for this phone.", !this.inviteValue, "Turn off")
      }
    } catch {
      this.show("Notifications could not be turned on. Allow them for this site and try again.", true, "Try again")
    } finally {
      this.buttonTarget.disabled = false
    }
  }

  async save(subscription) {
    const response = await fetch("/household/push_subscription", { method: "POST", headers: this.headers(), body: JSON.stringify({ subscription: subscription.toJSON() }) })
    if (!response.ok) throw new Error("save failed")
  }

  headers() {
    return { "Content-Type": "application/json", "X-CSRF-Token": document.querySelector("meta[name=csrf-token]")?.content }
  }

  show(message, withButton, label) {
    this.element.hidden = false
    this.messageTarget.textContent = message
    this.buttonTarget.hidden = !withButton
    if (label) this.buttonTarget.textContent = label
  }

  decode(base64) {
    const padded = (base64 + "=".repeat((4 - base64.length % 4) % 4)).replace(/-/g, "+").replace(/_/g, "/")
    return Uint8Array.from(atob(padded), char => char.charCodeAt(0))
  }
}
