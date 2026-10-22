import { Controller } from "@hotwired/stimulus"

// Records a voice note in the browser and puts it in the form's file field.
// Without MediaRecorder the plain "Audio file" picker stays visible instead.
export default class extends Controller {
  static targets = [ "start", "stop", "input", "fallback", "panel", "status", "preview", "dot" ]
  static values = { maxSeconds: { type: Number, default: 300 } }

  connect() {
    if (window.MediaRecorder && navigator.mediaDevices?.getUserMedia) {
      this.startTarget.hidden = false
      this.fallbackTarget.hidden = true
    }
  }

  disconnect() {
    this.stop()
  }

  async start() {
    this.panelTarget.hidden = false
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      this.statusTarget.textContent = "No access to the microphone"
      return
    }
    const type = [ "audio/webm;codecs=opus", "audio/mp4", "audio/ogg;codecs=opus" ].find(t => MediaRecorder.isTypeSupported(t))
    this.chunks = []
    this.recorder = new MediaRecorder(this.stream, type ? { mimeType: type, audioBitsPerSecond: 32000 } : {})
    this.recorder.ondataavailable = event => { if (event.data.size) this.chunks.push(event.data) }
    this.recorder.onstop = () => this.finish()
    this.recorder.start()

    this.startedAt = Date.now()
    this.startTarget.hidden = true
    this.stopTarget.hidden = false
    this.dotTarget.hidden = false
    this.previewTarget.hidden = true
    this.tick()
    this.timer = setInterval(() => this.tick(), 500)
  }

  stop() {
    clearInterval(this.timer)
    if (this.recorder?.state === "recording") this.recorder.stop()
    this.stream?.getTracks().forEach(track => track.stop())
  }

  discard() {
    this.discarding = true
    this.stop()
    this.inputTarget.value = ""
    this.previewTarget.removeAttribute("src")
    this.panelTarget.hidden = true
    this.stopTarget.hidden = true
    this.startTarget.hidden = false
  }

  tick() {
    const seconds = Math.floor((Date.now() - this.startedAt) / 1000)
    this.statusTarget.textContent = `Recording… ${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`
    if (seconds >= this.maxSecondsValue) this.stop()
  }

  finish() {
    this.stopTarget.hidden = true
    this.startTarget.hidden = false
    this.dotTarget.hidden = true
    if (this.discarding) { this.discarding = false; return }

    const type = (this.recorder.mimeType || "audio/webm").split(";")[0]
    const extension = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm"
    const file = new File(this.chunks, `voice-note.${extension}`, { type })
    const files = new DataTransfer()
    files.items.add(file)
    this.inputTarget.files = files.files

    this.statusTarget.textContent = "Voice note ready"
    this.previewTarget.src = URL.createObjectURL(file)
    this.previewTarget.hidden = false
  }
}
