from __future__ import annotations

import io
import logging
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Audio constants
SAMPLE_RATE = 16000   # Whisper expects 16kHz
CHANNELS = 1          # Mono audio
DTYPE = np.float32    # 32-bit float samples


class AudioRecorder:
    """Records audio from the microphone using sounddevice.

    Supports two modes:
    1. Push-to-talk: record while a key is held (uses keyboard events).
    2. Timed recording: record for a fixed duration.
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, channels: int = CHANNELS) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._stream: Any = None

    def record_seconds(self, duration: float = 5.0) -> np.ndarray:
        """Record audio for a fixed number of seconds.

        Args:
            duration: Recording duration in seconds.

        Returns:
            Numpy array of recorded audio samples (float32, mono, 16kHz).
        """
        import sounddevice as sd

        logger.info("Recording for %.1f seconds...", duration)
        num_samples = int(self.sample_rate * duration)
        audio = sd.rec(
            num_samples,
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
        )
        sd.wait()  # Block until recording is complete
        logger.info("Recording complete: %d samples", len(audio))
        return audio.flatten()

    def start_recording(self) -> None:
        """Start continuous recording (for push-to-talk)."""
        import sounddevice as sd

        self._frames = []
        self._recording = True

        def callback(indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
            if status:
                logger.warning("Audio input status: %s", status)
            if self._recording:
                self._frames.append(indata.copy())

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            callback=callback,
            blocksize=int(self.sample_rate * 0.1),  # 100ms blocks
        )
        self._stream.start()
        logger.info("Push-to-talk recording started")

    def stop_recording(self) -> np.ndarray:
        """Stop recording and return the captured audio.

        Returns:
            Numpy array of recorded audio samples.
        """
        self._recording = False
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if not self._frames:
            logger.warning("No audio frames captured")
            return np.array([], dtype=np.float32)

        audio = np.concatenate(self._frames).flatten()
        duration = len(audio) / self.sample_rate
        logger.info("Recording stopped: %.1fs, %d samples", duration, len(audio))
        self._frames = []
        return audio

    @staticmethod
    def save_wav(audio: np.ndarray, filepath: str | Path, sample_rate: int = SAMPLE_RATE) -> Path:
        """Save audio array as a WAV file (uses stdlib wave, no scipy needed)."""
        import struct
        import wave

        filepath = Path(filepath)
        # Convert float32 [-1.0, 1.0] to int16
        audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

        with wave.open(str(filepath), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit = 2 bytes
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        logger.info("Audio saved: %s", filepath)
        return filepath


class AudioPlayer:
    """Plays audio through the system speakers using pygame."""

    def __init__(self) -> None:
        self._initialized = False

    def _ensure_init(self) -> None:
        """Initialize pygame mixer on first use."""
        if not self._initialized:
            import pygame
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
            self._initialized = True
            logger.info("Audio player initialized (pygame mixer)")

    def play_bytes(self, audio_bytes: bytes, block: bool = True, filetype: str = "mp3") -> None:
        """Play raw audio bytes (MP3 format from ElevenLabs).

        Args:
            audio_bytes: Encoded audio data.
            block: If True, wait for playback to finish before returning.
            filetype: Audio format hint passed to pygame.
        """
        if not audio_bytes:
            logger.warning("Empty audio data, nothing to play")
            return

        import pygame
        self._ensure_init()

        logger.info("Playing audio: %d bytes", len(audio_bytes))
        audio_stream = io.BytesIO(audio_bytes)
        pygame.mixer.music.load(audio_stream, filetype)
        pygame.mixer.music.play()

        if block:
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)
            logger.info("Audio playback complete")

    def play_file(self, filepath: str | Path, block: bool = True) -> None:
        """Play an audio file.

        Args:
            filepath: Path to audio file.
            block: If True, wait for playback to finish.
        """
        import pygame
        self._ensure_init()

        logger.info("Playing file: %s", filepath)
        pygame.mixer.music.load(str(filepath))
        pygame.mixer.music.play()

        if block:
            while pygame.mixer.music.get_busy():
                pygame.time.wait(50)

    def stop(self) -> None:
        """Stop any currently playing audio."""
        try:
            import pygame
            if self._initialized:
                pygame.mixer.music.stop()
        except Exception:
            pass

    def cleanup(self) -> None:
        """Cleanup pygame mixer resources."""
        try:
            import pygame
            if self._initialized:
                pygame.mixer.quit()
                self._initialized = False
        except Exception:
            pass
