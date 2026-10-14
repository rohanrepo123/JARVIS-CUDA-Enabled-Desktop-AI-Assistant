from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True, slots=True)
class JarvisConfig:
    """Central configuration for the JARVIS voice agent.

    Loaded from environment variables (with .env file support).
    All sensitive values are excluded from repr for security.
    """

    # ── Ollama (Local LLM) ───────────────────────────────────────────
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"

    # ── ElevenLabs (Text-to-Speech) ──────────────────────────────────
    elevenlabs_api_key: str = field(default="", repr=False)
    elevenlabs_voice_id: str = "JBFqnCBsd6RMkjVDRZzb"

    # ── Whisper (Speech-to-Text) ─────────────────────────────────────
    whisper_model_size: str = "base"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"

    # ── Interaction Mode ─────────────────────────────────────────────
    mode: str = "text"  # 'text' or 'voice'

    # ── Server ───────────────────────────────────────────────────────
    host: str = "127.0.0.1"
    port: int = 8000

    # ── Storage ──────────────────────────────────────────────────────
    database_path: str = ""

    @classmethod
    def from_environment(cls) -> JarvisConfig:
        """Load configuration from environment variables and .env file."""
        load_dotenv()

        # Resolve database path
        db_path = os.getenv("JARVIS_DATABASE_PATH", "")
        if not db_path:
            local_data = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
            db_path = str(local_data / "JARVIS" / "jarvis.db")

        return cls(
            ollama_url=os.getenv("JARVIS_OLLAMA_URL", "http://127.0.0.1:11434"),
            ollama_model=os.getenv("JARVIS_OLLAMA_MODEL", "llama3.2"),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "JBFqnCBsd6RMkjVDRZzb"),
            whisper_model_size=os.getenv("WHISPER_MODEL_SIZE", "base"),
            whisper_device=os.getenv("WHISPER_DEVICE", "cpu"),
            whisper_compute_type=os.getenv("WHISPER_COMPUTE_TYPE", "int8"),
            mode=os.getenv("JARVIS_MODE", "text"),
            host=os.getenv("JARVIS_HOST", "127.0.0.1"),
            port=int(os.getenv("JARVIS_PORT", "8000")),
            database_path=db_path,
        )

    @property
    def voice_enabled(self) -> bool:
        """Check if voice mode can be activated (needs ElevenLabs API key)."""
        return bool(self.elevenlabs_api_key and self.elevenlabs_api_key != "your_elevenlabs_api_key_here")

    def __str__(self) -> str:
        return (
            f"JarvisConfig(\n"
            f"  mode={self.mode},\n"
            f"  ollama={self.ollama_url} [{self.ollama_model}],\n"
            f"  whisper={self.whisper_model_size} on {self.whisper_device},\n"
            f"  elevenlabs={'configured' if self.voice_enabled else 'not configured'},\n"
            f"  voice_id={self.elevenlabs_voice_id},\n"
            f")"
        )
