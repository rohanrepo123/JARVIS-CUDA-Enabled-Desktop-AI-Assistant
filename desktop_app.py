"""Windows desktop interface for the JARVIS assistant."""
from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import tkinter as tk
from tkinter import messagebox, ttk

# faster-whisper uses regular files when Windows symlinks are unavailable.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


class JarvisDesktop:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("JARVIS — Local AI Assistant")
        self.root.geometry("900x700")
        self.root.minsize(620, 500)
        self.root.configure(bg="#101827")
        self.pipeline = None
        self.config = None
        self.busy = False
        self.recording = False
        self.closing = False
        self.settings_path = Path(__file__).parent / "data" / "desktop-settings.json"
        self.settings = {"speak_voice_replies": True}
        try:
            self.settings.update(json.loads(self.settings_path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
        self._build_ui()
        self._append("JARVIS", "Starting assistant… Make sure Ollama is running.")
        threading.Thread(target=self._initialize, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=8)

        header_row = tk.Frame(self.root, bg="#101827")
        header_row.pack(fill="x", padx=18, pady=(8, 4))
        tk.Label(header_row, text="J.A.R.V.I.S.", bg="#101827", fg="#5ee7ff",
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        ttk.Button(header_row, text="New chat", command=self._new_conversation).pack(side="right", padx=(8, 0))
        ttk.Button(header_row, text="Settings", command=self._open_settings).pack(side="right")

        self.chat = tk.Text(self.root, wrap="word", state="disabled", bg="#172235",
                            fg="#e6edf7", insertbackground="white", relief="flat",
                            padx=16, pady=12, font=("Segoe UI", 11), spacing3=8)
        self.chat.pack(fill="both", expand=True, padx=16, pady=(4, 10))
        self.chat.tag_configure("you", foreground="#74d6ff", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("jarvis", foreground="#80f0bb", font=("Segoe UI", 10, "bold"))
        self.chat.tag_configure("body", foreground="#e6edf7")

        row = tk.Frame(self.root, bg="#101827")
        row.pack(fill="x", padx=16, pady=(0, 8))
        self.entry = tk.Entry(row, bg="#172235", fg="white", insertbackground="white",
                              relief="flat", font=("Segoe UI", 11))
        self.entry.pack(side="left", fill="x", expand=True, ipady=11, padx=(0, 8))
        self.entry.bind("<Return>", lambda _event: self._send())
        self.send_button = ttk.Button(row, text="Send", command=self._send, state="disabled")
        self.send_button.pack(side="right")

        voice_row = tk.Frame(self.root, bg="#101827")
        voice_row.pack(fill="x", padx=16, pady=(0, 8))
        self.voice_button = ttk.Button(voice_row, text="Hold to Talk", state="disabled")
        self.voice_button.pack(side="left")
        self.voice_button.bind("<ButtonPress-1>", self._start_voice)
        self.voice_button.bind("<ButtonRelease-1>", self._stop_voice)
        self.voice_hint = tk.Label(voice_row, text="Hold the button while speaking",
                                   bg="#101827", fg="#9aabc2", font=("Segoe UI", 9))
        self.voice_hint.pack(side="left", padx=10)

        self.status = tk.Label(self.root, text="Connecting…", bg="#101827", fg="#9aabc2",
                               anchor="w", font=("Segoe UI", 9), padx=18, pady=7)
        self.status.pack(fill="x")

    def _append(self, who: str, text: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{who}\n", "you" if who.startswith("You") else "jarvis")
        self.chat.insert("end", f"{text}\n\n", "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _start_streamed_reply(self) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", "JARVIS\n", "jarvis")
        self.chat.configure(state="disabled")
        self.chat.see("end")
        self.reply_chunks = 0

    def _stream_chunk(self, chunk: str) -> None:
        if self.closing:
            return
        self.chat.configure(state="normal")
        self.chat.insert("end", chunk, "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")
        self.reply_chunks += len(chunk)

    def _finish_streamed_reply(self, answer: str) -> None:
        # The model may return a non-streaming fallback for an empty response.
        if not self.reply_chunks and answer:
            self._stream_chunk(answer)
        self.chat.configure(state="normal")
        self.chat.insert("end", "\n\n", "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _initialize(self) -> None:
        try:
            from run import create_pipeline
            from backend.app.config import JarvisConfig
            config = JarvisConfig.from_environment()
            pipeline = create_pipeline(config)
            from backend.app.voice.tts import create_tts_provider
            provider = self.settings.get("tts_provider", config.tts_provider)
            try:
                pipeline.tts = create_tts_provider(
                    provider,
                    api_key=config.elevenlabs_api_key,
                    voice_id=config.elevenlabs_voice_id,
                )
            except ValueError:
                provider = "kokoro"
                pipeline.tts = create_tts_provider(provider)
                self.settings["tts_provider"] = provider
            self.pipeline = pipeline
            self.config = config
            self.root.after(0, self._ready)
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._startup_failed(error))

    def _ready(self) -> None:
        self.send_button.configure(state="normal")
        self.voice_button.configure(state="normal" if self.pipeline and self.pipeline.voice_available else "disabled")
        model = self.config.ollama_model if self.config else "Ollama"
        self.status.configure(text=f"Ready · {model} · {self._tts_provider_label()}")
        self._append("JARVIS", "I'm online. Type a message or hold the microphone button to speak.")
        self.entry.focus_set()

    def _startup_failed(self, error: str) -> None:
        self.status.configure(text="Could not start")
        self._append("JARVIS", f"Startup failed: {error}\nCheck that Ollama is running and configured.")

    def _send(self) -> None:
        message = self.entry.get().strip()
        if not message or self.busy or self.pipeline is None:
            return
        self.entry.delete(0, "end")
        self._append("You", message)
        self._begin_response("Thinking…")
        threading.Thread(target=self._respond, args=(message,), daemon=True).start()

    def _begin_response(self, status: str) -> None:
        self.busy = True
        self.send_button.configure(state="disabled")
        self.voice_button.configure(state="disabled")
        self.status.configure(text=status)
        self._start_streamed_reply()

    def _respond(self, message: str) -> None:
        try:
            answer = self.pipeline.process_text_stream(
                message, lambda chunk: self.root.after(0, self._stream_chunk, chunk), mode="text"
            )
            self.root.after(0, self._finish_streamed_reply, answer)
        except Exception as exc:
            self.root.after(0, self._finish_streamed_reply, f"Sorry, I hit an error: {exc}")
        finally:
            self.root.after(0, self._response_done)

    def _start_voice(self, _event=None) -> None:
        if self.busy or self.pipeline is None or not self.pipeline.voice_available:
            return
        try:
            self.pipeline.start_listening()
            self.recording = True
            self.busy = True
            self.send_button.configure(state="disabled")
            self.voice_button.configure(text="Listening…", state="normal")
            self.status.configure(text="Listening · release to send")
        except Exception as exc:
            self._append("JARVIS", f"Could not start microphone: {exc}")
            self._response_done()

    def _stop_voice(self, _event=None) -> None:
        if not self.recording:
            return
        self.recording = False
        self.voice_button.configure(text="Hold to Talk", state="disabled")
        self.status.configure(text="Transcribing…")
        threading.Thread(target=self._voice_turn, daemon=True).start()

    def _voice_turn(self) -> None:
        try:
            audio = self.pipeline.recorder.stop_recording()
            if len(audio) < self.pipeline.recorder.sample_rate * 0.3:
                self.root.after(0, self._append, "JARVIS", "That recording was too short. Please try again.")
                return
            transcript = self.pipeline.stt.transcribe_array(audio)
            if not transcript.strip():
                self.root.after(0, self._append, "JARVIS", "I couldn't understand that. Please try again.")
                return
            self.root.after(0, self._append, "You · voice", transcript)
            self.root.after(0, self._start_streamed_reply)
            self.root.after(0, lambda: self.status.configure(text="Thinking…"))
            answer = self.pipeline.process_text_stream(
                transcript,
                lambda chunk: self.root.after(0, self._stream_chunk, chunk),
                mode="voice",
            )
            self.root.after(0, self._finish_streamed_reply, answer)
            if self.settings["speak_voice_replies"] and self.pipeline.tts:
                self.pipeline._speak(answer)
        except Exception as exc:
            self.root.after(0, self._append, "JARVIS", f"Voice input failed: {exc}")
        finally:
            self.root.after(0, self._response_done)

    def _response_done(self) -> None:
        if self.closing:
            return
        self.busy = False
        self.recording = False
        self.send_button.configure(state="normal" if self.pipeline else "disabled")
        self.voice_button.configure(
            text="Hold to Talk",
            state="normal" if self.pipeline and self.pipeline.voice_available else "disabled",
        )
        model = self.config.ollama_model if self.config else "local Ollama"
        self.status.configure(text=f"Ready · {model} · {self._tts_provider_label()}")
        self.entry.focus_set()

    def _tts_provider_label(self) -> str:
        provider = self.settings.get(
            "tts_provider", self.config.tts_provider if self.config else "kokoro"
        )
        return "Kokoro TTS" if provider == "kokoro" else "ElevenLabs TTS"

    def _new_conversation(self) -> None:
        if self.busy or self.pipeline is None:
            return
        self.pipeline.new_conversation()
        self.chat.configure(state="normal")
        self.chat.delete("1.0", "end")
        self.chat.configure(state="disabled")
        self._append("JARVIS", "A fresh conversation is ready. What would you like to work on?")

    def _open_settings(self) -> None:
        dialog = tk.Toplevel(self.root)
        dialog.title("JARVIS Settings")
        dialog.configure(bg="#101827")
        dialog.resizable(False, False)
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(dialog, text="Settings", bg="#101827", fg="#5ee7ff",
                 font=("Segoe UI", 16, "bold")).pack(anchor="w", padx=18, pady=(16, 8))
        model = self.config.ollama_model if self.config else "Loading…"
        whisper = self.config.whisper_model_size if self.config else "Loading…"
        device = self.config.whisper_device.upper() if self.config else "—"
        tk.Label(dialog, text=f"Chat model: {model}\nSpeech recognition: Whisper {whisper} ({device})",
                 bg="#101827", fg="#e6edf7", justify="left", font=("Segoe UI", 10)).pack(anchor="w", padx=18, pady=8)
        provider_labels = {
            "kokoro": "Kokoro · local CUDA",
            "elevenlabs": "ElevenLabs · cloud",
        }
        provider_var = tk.StringVar(
            value=provider_labels.get(
                self.settings.get(
                    "tts_provider", self.config.tts_provider if self.config else "kokoro"
                ),
                provider_labels["kokoro"],
            )
        )
        tk.Label(dialog, text="Speech engine", bg="#101827", fg="#e6edf7",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=18, pady=(10, 3))
        provider_menu = ttk.Combobox(
            dialog,
            textvariable=provider_var,
            values=list(provider_labels.values()),
            state="readonly",
            width=30,
        )
        provider_menu.pack(anchor="w", padx=18)
        tk.Label(
            dialog,
            text="Kokoro runs on this device and downloads its model on first use. ElevenLabs requires an API key.",
            bg="#101827", fg="#9aabc2", font=("Segoe UI", 9), wraplength=360, justify="left",
        ).pack(anchor="w", padx=18, pady=(5, 4))
        speak_var = tk.BooleanVar(value=self.settings["speak_voice_replies"])
        speak_toggle = ttk.Checkbutton(dialog, text="Speak replies to voice messages",
                                       variable=speak_var, state="normal")
        speak_toggle.pack(anchor="w", padx=18, pady=(8, 2))

        def save() -> None:
            provider = next(
                key for key, label in provider_labels.items() if label == provider_var.get()
            )
            if self.pipeline is not None and self.config is not None:
                try:
                    from backend.app.voice.tts import create_tts_provider
                    self.pipeline.tts = create_tts_provider(
                        provider,
                        api_key=self.config.elevenlabs_api_key,
                        voice_id=self.config.elevenlabs_voice_id,
                    )
                except ValueError as exc:
                    messagebox.showerror("Speech engine", str(exc), parent=dialog)
                    return
            self.settings["tts_provider"] = provider
            self.settings["speak_voice_replies"] = speak_var.get()
            try:
                self.settings_path.parent.mkdir(parents=True, exist_ok=True)
                self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
            except OSError:
                messagebox.showwarning("Settings", "Could not save settings on this device.", parent=dialog)
            dialog.destroy()

        ttk.Button(dialog, text="Done", command=save).pack(anchor="e", padx=18, pady=16)

    def _close(self) -> None:
        self.closing = True
        if self.pipeline:
            if self.recording:
                try:
                    self.pipeline.recorder.stop_recording()
                except Exception:
                    pass
            threading.Thread(target=self.pipeline.cleanup, daemon=True).start()
        self.root.destroy()


if __name__ == "__main__":
    app = tk.Tk()
    JarvisDesktop(app)
    app.mainloop()
