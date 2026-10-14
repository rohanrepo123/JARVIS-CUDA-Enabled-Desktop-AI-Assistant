from __future__ import annotations

import logging
import time
from typing import Any, Callable

from backend.app.agent.graph import build_jarvis_graph
from backend.app.voice.audio import AudioPlayer, AudioRecorder
from backend.app.voice.stt import WhisperSTT
from backend.app.voice.tts import ElevenLabsTTS

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
        tts: ElevenLabsTTS | None = None,
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

        result: dict[str, Any] = {}
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
                        on_chunk(content)
            elif stream_kind == "updates" and "respond" in payload:
                result = payload["respond"]
        return str(result.get("response", "I couldn't generate a response."))

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

        # Step 3: Process through LangGraph agent (voice mode for natural speech)
        response = self.process_text(transcript, mode="voice", speak=False)

        # Step 4: Speak the response
        self._speak(response)

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
        response = self.process_text(transcript, mode="voice", speak=False)
        self._speak(response)
        return response

    def start_listening(self) -> None:
        """Start recording audio (for push-to-talk key press)."""
        self.recorder.start_recording()

    def _speak(self, text: str) -> None:
        """Synthesize and play speech."""
        if self.tts is None:
            logger.warning("TTS not available, skipping speech")
            return

        try:
            audio_bytes = self.tts.synthesize(text)
            if audio_bytes:
                self.player.play_bytes(
                    audio_bytes,
                    block=True,
                    filetype=getattr(self.tts, "audio_format", "mp3"),
                )
        except Exception as exc:
            logger.error("TTS/playback failed: %s", exc)

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
