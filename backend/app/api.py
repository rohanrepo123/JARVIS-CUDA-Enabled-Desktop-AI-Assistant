from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ── Request / Response Models ────────────────────────────────────────

class ChatRequest(BaseModel):
    """Request body for the chat endpoint."""
    message: str = Field(min_length=1, max_length=12_000, description="User's message")
    mode: str = Field(default="text", pattern=r"^(text|voice)$", description="Interaction mode")
    speak: bool = Field(default=False, description="Whether to speak the response (voice mode)")


class ChatResponse(BaseModel):
    """Response from the chat endpoint."""
    response: str
    mode: str
    spoken: bool = False


class ModeRequest(BaseModel):
    """Request to switch interaction mode."""
    mode: str = Field(pattern=r"^(text|voice)$")


class StatusResponse(BaseModel):
    """System status response."""
    status: str
    mode: str
    model: str
    voice_available: bool
    stt_loaded: bool
    tts_configured: bool


# ── API Routes ───────────────────────────────────────────────────────

def create_router() -> APIRouter:
    """Create and return the JARVIS API router."""
    router = APIRouter(prefix="/api", tags=["JARVIS"])

    @router.get("/health")
    async def health() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "online", "service": "JARVIS"}

    @router.get("/status", response_model=StatusResponse)
    async def status(request: Request) -> StatusResponse:
        """Get current JARVIS system status."""
        config = request.app.state.config
        pipeline = request.app.state.pipeline
        return StatusResponse(
            status="online",
            mode=config.mode,
            model=config.ollama_model,
            voice_available=pipeline.voice_available,
            stt_loaded=pipeline.stt is not None and pipeline.stt._model is not None,
            tts_configured=pipeline.tts is not None,
        )

    @router.post("/chat", response_model=ChatResponse)
    async def chat(body: ChatRequest, request: Request) -> ChatResponse:
        """Send a message to JARVIS and get a response.

        If mode is 'voice' and speak is True, JARVIS will also speak the response.
        """
        pipeline = request.app.state.pipeline
        config = request.app.state.config

        # Determine effective mode
        effective_mode = body.mode or config.mode
        should_speak = body.speak and pipeline.voice_available and effective_mode == "voice"

        try:
            response_text = pipeline.process_text(
                text=body.message,
                mode=effective_mode,
                speak=should_speak,
            )
            return ChatResponse(
                response=response_text,
                mode=effective_mode,
                spoken=should_speak,
            )
        except Exception as exc:
            logger.exception("Chat processing failed")
            raise HTTPException(status_code=500, detail=f"Processing failed: {exc}") from exc

    @router.post("/voice/record")
    async def voice_record(request: Request) -> ChatResponse:
        """Record audio, transcribe, process, and speak response.

        This is the full voice interaction endpoint.
        """
        pipeline = request.app.state.pipeline

        if not pipeline.voice_available:
            raise HTTPException(
                status_code=503,
                detail="Voice mode unavailable. Configure ELEVENLABS_API_KEY in .env",
            )

        try:
            response_text = pipeline.process_voice(record_seconds=5.0)
            return ChatResponse(response=response_text, mode="voice", spoken=True)
        except Exception as exc:
            logger.exception("Voice processing failed")
            raise HTTPException(status_code=500, detail=f"Voice processing failed: {exc}") from exc

    @router.post("/mode")
    async def set_mode(body: ModeRequest, request: Request) -> dict[str, Any]:
        """Switch between text and voice mode."""
        config = request.app.state.config
        pipeline = request.app.state.pipeline

        if body.mode == "voice" and not pipeline.voice_available:
            raise HTTPException(
                status_code=400,
                detail="Cannot switch to voice mode: ElevenLabs API key not configured.",
            )

        # Update the config mode (mutable override via app state)
        request.app.state.current_mode = body.mode
        logger.info("Mode switched to: %s", body.mode)
        return {"mode": body.mode, "voice_available": pipeline.voice_available}

    @router.post("/conversation/new")
    async def new_conversation(request: Request) -> dict[str, str]:
        """Start a new conversation thread."""
        pipeline = request.app.state.pipeline
        pipeline.new_conversation()
        return {"thread_id": pipeline.thread_id, "status": "created"}

    return router
