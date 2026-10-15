from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Callable

from backend.app.agent.graph import build_jarvis_graph
from backend.app.agent.response_format import AssistantResponseStream
from backend.app.voice.audio import AudioPlayer, AudioRecorder
from backend.app.voice.stt import WhisperSTT
from backend.app.voice.tts import ElevenLabsTTS, KokoroTTS

logger = logging.getLogger(__name__)


class VoicePipeline:
    """Full voice interaction pipeline: Mic -> Whisper -> LangGraph -> ElevenLabs -> Speaker.

    Orchestrates the complete voice conversation loop. When mode is 'voice',
    the pipeline records audio, transcribes it, processes through the LangGraph
    agent, synthesizes speech, and plays it back. When mode is 'text', it
    just processes text through the agent and optionally speaks the response.
    """

    def __init__(
        self,
        graph: Any,
        stt: WhisperSTT | None = None,
        tts: ElevenLabsTTS | KokoroTTS | None = None,
        recorder: AudioRecorder | None = None,
        player: AudioPlayer | None = None,
        thread_id: str = "jarvis-main",
    ) -> None:
        self.graph = graph
        self.stt = stt
        self.tts = tts
        self.recorder = recorder or AudioRecorder()
        self.player = player or AudioPlayer()
        self.thread_id = thread_id
        self._config = {"configurable": {"thread_id": thread_id}}

    @property
    def voice_available(self) -> bool:
        """Check whether local speech recognition is available."""
        return self.stt is not None

    def process_text(self, text: str, mode: str = "text", speak: bool = False) -> str:
        """Process a text input through the agent and optionally speak the response.

        Args:
            text: User's text input.
            mode: 'text' or 'voice' — affects how the LLM formats its response.
            speak: If True and TTS is available, speak the response aloud.

        Returns:
            The agent's response text.
        """
        if not text.strip():
            return "I didn't catch anything. Could you try again?"

        if mode == "voice":
            spoken, details = self.process_voice_turn(text, speak=speak)
            return f"{spoken}\n\n{details}" if details else spoken

        logger.info("Processing text input: '%s'", text[:80])
        start = time.monotonic()

        # Invoke the LangGraph agent
        result = self.graph.invoke(
            {
                "current_input": text,
                "mode": mode,
                "response": "",
                "tool_results": [],
            },
            config=self._config,
        )

        response = result.get("response", "I couldn't generate a response.")
        elapsed = time.monotonic() - start
        logger.info("Agent responded in %.1fs: '%s'", elapsed, response[:80])

        # Speak the response if requested and TTS is available
        if speak and self.tts is not None:
            self._speak(response)

        return response

    def process_text_stream(
        self, text: str, on_chunk: Callable[[str], None], mode: str = "text"
    ) -> str:
        """Run a turn and report assistant text as Ollama generates it."""
        if not text.strip():
            return "I didn't catch anything. Could you try again?"

        if mode == "voice":
            spoken, details = self.process_voice_turn(text, speak=False)
            answer = f"{spoken}\n\nDetails\n{details}" if details else spoken
            on_chunk(answer)
            return answer

        result: dict[str, Any] = {}
        defer_stream = _needs_verified_productivity_result(text)
        response_stream = AssistantResponseStream(
            on_summary=(lambda _chunk: None) if defer_stream else on_chunk,
            on_details_start=(lambda: None) if defer_stream else lambda: on_chunk("\n\nDetails\n"),
            on_details=(lambda _chunk: None) if defer_stream else on_chunk,
        )
        for stream_kind, payload in self.graph.stream(
            {
                "current_input": text,
                "mode": mode,
                "response": "",
                "tool_results": [],
            },
            config=self._config,
            stream_mode=["messages", "updates"],
        ):
            if stream_kind == "messages":
                message, metadata = payload
                if metadata.get("langgraph_node") == "agent":
                    content = getattr(message, "content", "")
                    if isinstance(content, str) and content:
                        response_stream.feed(content)
            elif stream_kind == "updates" and "respond" in payload:
                result = payload["respond"]
        response_stream.finish()
        final_response = str(result.get("response", "I couldn't generate a response."))
        if defer_stream:
            # Action answers are shown only after the tool result has passed
            # format_response's success check; this prevents live false claims.
            on_chunk(final_response)
            return final_response
        if not response_stream.valid and not response_stream.passthrough:
            on_chunk(final_response)
        return final_response

    def process_voice(self, record_seconds: float = 5.0) -> str:
        """Full voice interaction: record -> transcribe -> think -> speak.

        Args:
            record_seconds: How long to record (used for timed recording mode).

        Returns:
            The agent's response text.
        """
        if not self.voice_available:
            return "Voice mode is not available. Please configure STT and TTS."

        # Step 1: Record audio
        logger.info("=== Voice Turn Start ===")
        audio = self.recorder.record_seconds(record_seconds)

        if len(audio) < self.recorder.sample_rate * 0.3:  # Less than 0.3s
            return "Recording was too short. Please try again."

        # Step 2: Transcribe with Whisper
        transcript = self.stt.transcribe_array(audio)
        if not transcript.strip():
            return "I couldn't understand what you said. Please try again."

        logger.info("Transcript: '%s'", transcript)

        # Step 3: Speak a concise summary while the display details are generated.
        spoken, details = self.process_voice_turn(transcript, speak=True)
        response = f"{spoken}\n\n{details}" if details else spoken

        logger.info("=== Voice Turn Complete ===")
        return response

    def process_push_to_talk(self) -> str:
        """Push-to-talk voice interaction.

        Call start_listening() when the key is pressed, and
        stop_listening_and_process() when released.

        Returns:
            The agent's response text.
        """
        if not self.voice_available:
            return "Voice mode is not available."

        audio = self.recorder.stop_recording()

        if len(audio) < self.recorder.sample_rate * 0.3:
            return "Recording was too short."

        transcript = self.stt.transcribe_array(audio)
        if not transcript.strip():
            return "I couldn't understand that."

        logger.info("PTT Transcript: '%s'", transcript)
        spoken, details = self.process_voice_turn(transcript, speak=True)
        return f"{spoken}\n\n{details}" if details else spoken

    def start_listening(self) -> None:
        """Start recording audio (for push-to-talk key press)."""
        self.recorder.start_recording()

    def process_voice_turn(
        self,
        text: str,
        *,
        speak: bool = True,
        on_speech_started: Callable[[], None] | None = None,
    ) -> tuple[str, str]:
        """Generate separate speech and display answers, overlapping TTS with details."""
        if not text.strip():
            return "I didn't catch anything. Could you try again?", ""

        result: dict[str, Any] = {}
        speech_thread: threading.Thread | None = None
        speech_started = False
        defer_speech = _needs_verified_productivity_result(text)

        def start_speech(summary: str) -> None:
            nonlocal speech_thread, speech_started
            if speech_started or not summary.strip():
                return
            speech_started = True
            if speak and self.tts is not None:
                if on_speech_started:
                    on_speech_started()
                speech_thread = threading.Thread(
                    target=self._speak,
                    args=(summary,),
                    name="jarvis-voice-reply",
                    daemon=True,
                )
                speech_thread.start()

        response_stream = AssistantResponseStream(
            on_summary=lambda _chunk: None,
            on_summary_complete=(
                (lambda: None)
                if defer_speech
                else lambda: start_speech(response_stream.summary)
            ),
        )

        try:
            for stream_kind, payload in self.graph.stream(
                {
                    "current_input": text,
                    "mode": "voice",
                    "response": "",
                    "summary": "",
                    "display_details": "",
                    "is_detail_insight_needed": False,
                    "tool_results": [],
                },
                config=self._config,
                stream_mode=["messages", "updates"],
            ):
                if stream_kind == "messages":
                    message, metadata = payload
                    if metadata.get("langgraph_node") != "agent":
                        continue
                    content = getattr(message, "content", "")
                    if not isinstance(content, str) or not content:
                        continue
                    response_stream.feed(content)
                elif stream_kind == "updates" and "respond" in payload:
                    result = payload["respond"]
            response_stream.finish()
        finally:
            if speech_thread is not None:
                speech_thread.join()

        if not defer_speech and response_stream.valid and response_stream.summary_complete:
            spoken = response_stream.summary
            details = response_stream.details if response_stream.is_detail_insight_needed else ""
        else:
            spoken = str(result.get("summary", "")).strip()
            details = str(result.get("display_details", "")).strip()
        if not spoken:
            response = str(result.get("response", "I couldn't generate a response.")).strip()
            spoken = self._speech_summary(response)
            details = ""
        if not speech_started:
            start_speech(spoken)
        if speech_thread is not None:
            speech_thread.join()
        return spoken, details

    def _speak(self, text: str) -> None:
        """Synthesize and play speech."""
        if self.tts is None:
            logger.warning("TTS not available, skipping speech")
            return

        try:
            audio_bytes = self.tts.synthesize(self._speech_summary(text))
            if audio_bytes:
                self.player.play_bytes(
                    audio_bytes,
                    block=True,
                    filetype=getattr(self.tts, "audio_format", "mp3"),
                )
        except Exception as exc:
            logger.error("TTS/playback failed: %s", exc)

    @staticmethod
    def _speech_summary(text: str, max_chars: int = 420) -> str:
        """Keep spoken replies brief while preserving the full text reply."""
        clean = " ".join(text.split())
        if len(clean) <= max_chars:
            return clean
        excerpt = clean[:max_chars]
        boundary = max(excerpt.rfind(". "), excerpt.rfind("? "), excerpt.rfind("! "))
        excerpt = excerpt[: boundary + 1] if boundary >= 180 else excerpt.rsplit(" ", 1)[0]
        return f"{excerpt} For the full details, please see the text response."

    def new_conversation(self, thread_id: str | None = None) -> None:
        """Start a new conversation thread."""
        import uuid
        self.thread_id = thread_id or str(uuid.uuid4())
        self._config = {"configurable": {"thread_id": self.thread_id}}
        logger.info("New conversation: %s", self.thread_id)

    def cleanup(self) -> None:
        """Release all voice resources."""
        if self.stt:
            self.stt.unload()
        self.player.cleanup()
        logger.info("Voice pipeline cleaned up")


def _needs_verified_productivity_result(text: str) -> bool:
    """Avoid presenting action claims until a productivity tool result exists."""
    normalized = " ".join(text.lower().split())
    target = re.search(
        r"\b(?:alarm|reminders?|stopwatch)\b|\bremind\s+me\b|\bwake\s+me\s+up\b",
        normalized,
    )
    action = re.search(
        r"\b(?:set|create|add|schedule|cancel|start|begin|stop|pause|halt|reset|clear|remind me)\b"
        r"|\balarm\s+(?:for|at)\b|\bwake\s+me\s+up\b",
        normalized,
    )
    return bool(action and target)
