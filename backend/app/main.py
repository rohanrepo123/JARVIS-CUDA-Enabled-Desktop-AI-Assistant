from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import JarvisConfig

logger = logging.getLogger(__name__)


def create_app(config: JarvisConfig | None = None) -> FastAPI:
    """Create and configure the JARVIS FastAPI application."""
    config = config or JarvisConfig.from_environment()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Initialize resources on startup, cleanup on shutdown."""
        from backend.app.agent.graph import build_jarvis_graph
        from backend.app.voice.pipeline import VoicePipeline
        from backend.app.voice.stt import WhisperSTT
        from backend.app.voice.tts import ElevenLabsTTS
        from backend.app.voice.audio import AudioRecorder, AudioPlayer

        # Build the LangGraph agent brain
        logger.info("Initializing JARVIS brain...")
        graph = build_jarvis_graph(
            model=config.ollama_model,
            base_url=config.ollama_url,
        )

        # Initialize voice components (if configured)
        stt = None
        tts = None
        if config.voice_enabled:
            logger.info("Initializing voice pipeline...")
            stt = WhisperSTT(
                model_size=config.whisper_model_size,
                device=config.whisper_device,
                compute_type=config.whisper_compute_type,
            )
            tts = ElevenLabsTTS(
                api_key=config.elevenlabs_api_key,
                voice_id=config.elevenlabs_voice_id,
            )

        pipeline = VoicePipeline(
            graph=graph,
            stt=stt,
            tts=tts,
            recorder=AudioRecorder(),
            player=AudioPlayer(),
        )

        # Store in app state for API access
        app.state.config = config
        app.state.graph = graph
        app.state.pipeline = pipeline
        app.state.stt = stt
        app.state.tts = tts

        logger.info("JARVIS is online! Mode: %s", config.mode)
        try:
            yield
        finally:
            pipeline.cleanup()
            logger.info("JARVIS shutdown complete")

    app = FastAPI(
        title="JARVIS Voice Agent API",
        version="1.0.0",
        description="Local AI Assistant powered by LangGraph + Ollama + Whisper + ElevenLabs",
        lifespan=lifespan,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routes
    from backend.app.api import create_router
    app.include_router(create_router())

    return app


def run_server() -> None:
    """Start the JARVIS API server."""
    config = JarvisConfig.from_environment()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port, log_level="info")


if __name__ == "__main__":
    run_server()
