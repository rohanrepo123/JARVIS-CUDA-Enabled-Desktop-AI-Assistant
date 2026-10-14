"""Simple Windows desktop interface for the JARVIS voice agent."""
from __future__ import annotations

import os

# faster-whisper downloads public models from Hugging Face. On Windows without
# Developer Mode, the cache falls back to regular files; hide only that known
# non-fatal warning while preserving all other warnings.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import threading
import tkinter as tk
from tkinter import messagebox, ttk


class JarvisDesktop:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("JARVIS — Local AI Assistant")
        self.root.geometry("800x620")
        self.root.minsize(600, 450)
        self.root.configure(bg="#101827")
        self.pipeline = None
        self.busy = False
        self._build_ui()
        self._append("JARVIS", "Starting assistant… Make sure Ollama is running.")
        threading.Thread(target=self._initialize, daemon=True).start()
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", padding=8)
        header = tk.Label(self.root, text="J.A.R.V.I.S.", bg="#101827", fg="#5ee7ff",
                          font=("Segoe UI", 22, "bold"), pady=14)
        header.pack(fill="x")
        self.chat = tk.Text(self.root, wrap="word", state="disabled", bg="#172235",
                            fg="#e6edf7", insertbackground="white", relief="flat",
                            padx=16, pady=12, font=("Segoe UI", 11), spacing3=8)
        self.chat.pack(fill="both", expand=True, padx=16, pady=(0, 10))
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
        self.status = tk.Label(self.root, text="Connecting…", bg="#101827", fg="#9aabc2",
                               anchor="w", font=("Segoe UI", 9), padx=18, pady=7)
        self.status.pack(fill="x")

    def _append(self, who: str, text: str) -> None:
        self.chat.configure(state="normal")
        self.chat.insert("end", f"{who}\n", "you" if who == "You" else "jarvis")
        self.chat.insert("end", f"{text}\n\n", "body")
        self.chat.configure(state="disabled")
        self.chat.see("end")

    def _initialize(self) -> None:
        try:
            from run import create_pipeline
            from backend.app.config import JarvisConfig
            pipeline = create_pipeline(JarvisConfig.from_environment())
            self.pipeline = pipeline
            self.root.after(0, self._ready)
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._startup_failed(error))

    def _ready(self) -> None:
        self.send_button.configure(state="normal")
        self.status.configure(text="Ready · text chat · local Ollama")
        self._append("JARVIS", "I'm online. What can I help you with?")
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
        self.busy = True
        self.send_button.configure(state="disabled")
        self.status.configure(text="Thinking…")
        threading.Thread(target=self._respond, args=(message,), daemon=True).start()

    def _respond(self, message: str) -> None:
        try:
            answer = self.pipeline.process_text(message)
            self.root.after(0, lambda: self._append("JARVIS", answer))
        except Exception as exc:
            self.root.after(0, lambda error=str(exc): self._append("JARVIS", f"Sorry, I hit an error: {error}"))
        finally:
            self.root.after(0, self._response_done)

    def _response_done(self) -> None:
        self.busy = False
        self.send_button.configure(state="normal" if self.pipeline else "disabled")
        self.status.configure(text="Ready · text chat · local Ollama")
        self.entry.focus_set()

    def _close(self) -> None:
        if self.pipeline:
            threading.Thread(target=self.pipeline.cleanup, daemon=True).start()
        self.root.destroy()


if __name__ == "__main__":
    app = tk.Tk()
    JarvisDesktop(app)
    app.mainloop()
