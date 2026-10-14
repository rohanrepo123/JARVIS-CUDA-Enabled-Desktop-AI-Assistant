from __future__ import annotations

import logging
import os
import sysconfig
import time
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_CUDA_DLL_HANDLES: list[object] = []
_CUDA_DLL_PATHS_REGISTERED = False


def _register_nvidia_dll_directories() -> None:
    """Expose NVIDIA pip-wheel DLLs to CTranslate2 on Windows."""
    global _CUDA_DLL_PATHS_REGISTERED
    if _CUDA_DLL_PATHS_REGISTERED:
        return
    _CUDA_DLL_PATHS_REGISTERED = True

    site_packages = Path(sysconfig.get_paths()["purelib"])
    for relative in (
        Path("nvidia/cublas/bin"),
        Path("nvidia/cudnn/bin"),
        Path("nvidia/cuda_nvrtc/bin"),
    ):
        directory = site_packages / relative
        if not directory.is_dir():
            continue
        path = str(directory)
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
        if hasattr(os, "add_dll_directory"):
            _CUDA_DLL_HANDLES.append(os.add_dll_directory(path))


class WhisperSTT:
    """Speech-to-Text engine using faster-whisper (runs locally).

    Uses CTranslate2-backed Whisper models for fast, accurate transcription
    without sending any audio data to the cloud.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self._model: Any = None
        logger.info(
            "WhisperSTT configured: model=%s, device=%s, compute=%s",
            model_size, device, compute_type,
        )

    def _ensure_model(self) -> Any:
        """Lazy-load the Whisper model on first use to save startup time."""
        if self._model is None:
            _register_nvidia_dll_directories()
            from faster_whisper import WhisperModel
            logger.info("Loading Whisper model '%s' on %s...", self.model_size, self.device)
            start = time.monotonic()
            try:
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            except Exception as exc:
                if self.device.lower() not in {"cuda", "gpu"}:
                    raise
                logger.warning(
                    "Whisper GPU initialization failed (%s); retrying on CPU with int8.", exc
                )
                self.device = "cpu"
                self.compute_type = "int8"
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
            elapsed = time.monotonic() - start
            logger.info("Whisper model loaded in %.1fs", elapsed)
        return self._model

    def _transcribe(self, audio: str | Path | np.ndarray) -> tuple[str, Any]:
        """Transcribe audio, retrying on CPU if CUDA fails during inference."""
        model = self._ensure_model()
        try:
            segments, info = model.transcribe(
                audio,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            text = " ".join(segment.text for segment in segments).strip()
            return text, info
        except Exception as exc:
            if self.device.lower() not in {"cuda", "gpu"}:
                raise
            _register_nvidia_dll_directories()
            from faster_whisper import WhisperModel
            logger.warning(
                "Whisper GPU transcription failed (%s); retrying on CPU with int8.", exc
            )
            self._model = None
            self.device = "cpu"
            self.compute_type = "int8"
            model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
            self._model = model
            segments, info = model.transcribe(
                audio,
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
            )
            text = " ".join(segment.text for segment in segments).strip()
            return text, info

    def transcribe_file(self, audio_path: str | Path) -> str:
        """Transcribe an audio file to text.

        Args:
            audio_path: Path to a WAV/MP3/FLAC audio file.

        Returns:
            Transcribed text string.
        """
        logger.info("Transcribing file: %s", audio_path)
        start = time.monotonic()
        text, info = self._transcribe(str(audio_path))
        elapsed = time.monotonic() - start
        logger.info(
            "Transcription complete: %.1fs, lang=%s (prob=%.2f), text='%s'",
            elapsed, info.language, info.language_probability, text[:80],
        )
        return text

    def transcribe_array(self, audio_array: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a numpy audio array to text.

        Args:
            audio_array: Float32 numpy array of audio samples (mono, 16kHz recommended).
            sample_rate: Sample rate of the audio (default 16000 for Whisper).

        Returns:
            Transcribed text string.
        """
        logger.info("Transcribing audio array: %d samples, %.1fs duration",
                    len(audio_array), len(audio_array) / sample_rate)
        start = time.monotonic()

        # Ensure float32
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Normalize to [-1.0, 1.0] if needed
        if audio_array.max() > 1.0 or audio_array.min() < -1.0:
            audio_array = audio_array / max(abs(audio_array.max()), abs(audio_array.min()))

        text, info = self._transcribe(audio_array)
        elapsed = time.monotonic() - start
        logger.info("Transcription complete: %.1fs, text='%s'", elapsed, text[:80])
        return text

    def unload(self) -> None:
        """Unload the model to free memory (useful before loading LLM on GPU)."""
        if self._model is not None:
            del self._model
            self._model = None
            logger.info("Whisper model unloaded")
