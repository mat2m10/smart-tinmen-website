import { Controller } from "@hotwired/stimulus"

// Records a voice note in the browser and adds it to the form when it is sent
// (through the formdata event, so it doesn't depend on setting input.files).
// Posting while still recording stops first, then sends.
// Without MediaRecorder the plain "Audio file" picker stays visible instead.
export default class extends Controller {
  static targets = [ "start", "stop", "input", "fallback", "panel", "status", "preview", "dot" ]
  static values = { maxSeconds: { type: Number, default: 300 } }

  connect() {
    this.attach = this.attach.bind(this)
    this.element.addEventListener("formdata", this.attach)
    if (window.MediaRecorder && navigator.mediaDevices?.getUserMedia) {
      this.startTarget.hidden = false
      this.fallbackTarget.hidden = true
    }
  }

  disconnect() {
    this.element.removeEventListener("formdata", this.attach)
    this.stop()
  }

  // Posting while recording: stop, and send once the recording is ready.
  // A second tap while it is finishing waits for the same send.
  submit(event) {
    if (this.submitWhenReady) {
      event.preventDefault()
      event.stopImmediatePropagation()
      return
    }
    if (this.recorder?.state !== "recording") return
    event.preventDefault()
    event.stopImmediatePropagation()
    this.submitWhenReady = true
    this.statusTarget.textContent = "Finishing the recording…"
    this.stop()
  }

  attach(event) {
    const chosen = event.formData.get(this.inputTarget.name)
    if (this.file && !(chosen instanceof File && chosen.size > 0)) event.formData.set(this.inputTarget.name, this.file, this.file.name)
  }

  async start() {
    this.panelTarget.hidden = false
    this.statusTarget.textContent = "Starting the microphone…"
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: true })
    } catch {
      // Keep an earlier recording: it is still attached and still shown.
      this.statusTarget.textContent = this.file ? "No access to the microphone. The earlier voice note is kept." : "No access to the microphone"
      return
    }
    this.discarding = false
    this.file = null
    this.forgetPreview()
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

  // Throws away the recording, also one that is still being finished
  // (stop() was called but onstop has not fired yet). start() clears the flag.
  discard() {
    this.discarding = true
    this.submitWhenReady = false
    this.file = null
    this.stop()
    this.inputTarget.value = ""
    this.forgetPreview()
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

    const type = (this.recorder.mimeType || this.chunks[0]?.type || "audio/webm").split(";")[0]
    const extension = type.includes("mp4") ? "m4a" : type.includes("ogg") ? "ogg" : "webm"
    const file = new File(this.chunks, `voice-note.${extension}`, { type })
    if (file.size === 0) {
      this.submitWhenReady = false
      this.statusTarget.textContent = "Nothing was recorded. Check that the browser may use the microphone, then try again."
      return
    }
    this.file = file

    this.statusTarget.textContent = "Voice note ready"
    this.forgetPreview()
    this.previewTarget.src = URL.createObjectURL(file)
    this.previewTarget.hidden = false
    if (this.submitWhenReady) {
      this.submitWhenReady = false
      this.element.requestSubmit()
    }
  }

  forgetPreview() {
    if (this.previewTarget.src) URL.revokeObjectURL(this.previewTarget.src)
    this.previewTarget.removeAttribute("src")
  }
}
