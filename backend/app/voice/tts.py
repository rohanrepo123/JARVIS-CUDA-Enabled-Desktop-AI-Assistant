from __future__ import annotations

import io
import logging
import time
from typing import Any

logger = logging.getLogger(__name__)


class ElevenLabsTTS:
    """Text-to-Speech engine using ElevenLabs API.

    Converts text responses into natural-sounding speech audio.
    Supports both blocking and streaming synthesis.
    """

    def __init__(
        self,
        api_key: str,
        voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
        model_id: str = "eleven_flash_v2_5",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        style: float = 0.0,
    ) -> None:
        self.audio_format = "mp3"
        self.api_key = api_key
        self.voice_id = voice_id
        self.model_id = model_id
        self.voice_settings = {
            "stability": stability,
            "similarity_boost": similarity_boost,
            "style": style,
        }
        self._client: Any = None
        logger.info(
            "ElevenLabsTTS configured: voice=%s, model=%s",
            voice_id, model_id,
        )

    def _ensure_client(self) -> Any:
        """Lazy-initialize the ElevenLabs client."""
        if self._client is None:
            from elevenlabs import ElevenLabs
            self._client = ElevenLabs(api_key=self.api_key)
            logger.info("ElevenLabs client initialized")
        return self._client

    def synthesize(self, text: str) -> bytes:
        """Convert text to speech audio (blocking).

        Args:
            text: The text to convert to speech.

        Returns:
            Raw MP3 audio bytes.
        """
        if not text.strip():
            logger.warning("Empty text passed to TTS, skipping")
            return b""

        client = self._ensure_client()
        logger.info("Synthesizing speech: '%s' (%d chars)", text[:60], len(text))
        start = time.monotonic()

        # Generate audio using the ElevenLabs SDK
        audio_generator = client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
            voice_settings=self.voice_settings,
        )

        # Collect all chunks into bytes
        audio_bytes = b"".join(audio_generator)

        elapsed = time.monotonic() - start
        logger.info("TTS complete: %.1fs, %d bytes", elapsed, len(audio_bytes))
        return audio_bytes

    def synthesize_stream(self, text: str):
        """Stream audio chunks as they're generated (for lower latency).

        Args:
            text: The text to convert to speech.

        Yields:
            Chunks of MP3 audio bytes.
        """
        if not text.strip():
            return

        client = self._ensure_client()
        logger.info("Streaming TTS: '%s' (%d chars)", text[:60], len(text))

        audio_generator = client.text_to_speech.convert(
            voice_id=self.voice_id,
            text=text,
            model_id=self.model_id,
            voice_settings=self.voice_settings,
        )

        for chunk in audio_generator:
            if chunk:
                yield chunk

    def list_voices(self) -> list[dict[str, str]]:
        """List available ElevenLabs voices.

        Returns:
            List of voice info dicts with 'voice_id' and 'name'.
        """
        client = self._ensure_client()
        response = client.voices.get_all()
        return [
            {"voice_id": v.voice_id, "name": v.name}
            for v in response.voices
        ]


class KokoroTTS:
    """Local Kokoro text-to-speech with optional CUDA inference."""

    sample_rate = 24000
    audio_format = "wav"

    def __init__(
        self,
        device: str = "auto",
        lang_code: str = "a",
        voice: str = "af_heart",
        speed: float = 1.0,
    ) -> None:
        self.device = device
        self.lang_code = lang_code
        self.voice = voice
        self.speed = speed
        self._pipeline: Any = None

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            from kokoro import KPipeline

            device = None if self.device == "auto" else self.device
            logger.info("Loading Kokoro TTS on %s", device or "automatic device")
            self._pipeline = KPipeline(lang_code=self.lang_code, device=device)
        return self._pipeline

    def synthesize(self, text: str) -> bytes:
        """Synthesize text to an in-memory 24 kHz PCM WAV file."""
        if not text.strip():
            return b""

        import numpy as np
        import soundfile as sf

        pipeline = self._ensure_pipeline()
        chunks = []
        for _graphemes, _phonemes, audio in pipeline(text, voice=self.voice, speed=self.speed):
            if audio is not None:
                chunks.append(audio.detach().cpu().numpy())
        if not chunks:
            return b""

        audio = np.concatenate(chunks).astype(np.float32, copy=False)
        audio_buffer = io.BytesIO()
        sf.write(audio_buffer, audio, self.sample_rate, format="WAV", subtype="PCM_16")
        return audio_buffer.getvalue()


def create_tts_provider(
    provider: str,
    *,
    api_key: str = "",
    voice_id: str = "JBFqnCBsd6RMkjVDRZzb",
    kokoro_device: str = "auto",
) -> ElevenLabsTTS | KokoroTTS:
    """Create the selected TTS backend without loading local models yet."""
    if provider == "kokoro":
        return KokoroTTS(device=kokoro_device)
    if provider == "elevenlabs":
        if not api_key or api_key == "your_elevenlabs_api_key_here":
            raise ValueError("Set ELEVENLABS_API_KEY in .env to use ElevenLabs.")
        return ElevenLabsTTS(api_key=api_key, voice_id=voice_id)
    raise ValueError("TTS provider must be 'kokoro' or 'elevenlabs'.")
