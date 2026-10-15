#!/usr/bin/env python3
"""JARVIS Voice Agent — Interactive CLI.

Run this script to start JARVIS in your terminal.
Supports two modes:
  - TEXT mode:  Type messages, get text responses
  - VOICE mode: Speak to JARVIS, hear responses (requires ElevenLabs API key)

Usage:
    python run.py              # Start with mode from .env (default: text)
    python run.py --mode voice # Start in voice mode
    python run.py --mode text  # Start in text mode
"""
from __future__ import annotations

import argparse
import logging
import signal
import sys
import time

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

# ── Setup ────────────────────────────────────────────────────────────

console = Console()

JARVIS_BANNER = r"""
     ██╗ █████╗ ██████╗ ██╗   ██╗██╗███████╗
     ██║██╔══██╗██╔══██╗██║   ██║██║██╔════╝
     ██║███████║██████╔╝██║   ██║██║███████╗
██   ██║██╔══██║██╔══██╗╚██╗ ██╔╝██║╚════██║
╚█████╔╝██║  ██║██║  ██║ ╚████╔╝ ██║███████║
 ╚════╝ ╚═╝  ╚═╝╚═╝  ╚═╝  ╚═══╝  ╚═╝╚══════╝
"""


def setup_logging() -> None:
    """Configure logging for the CLI."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)-28s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler("jarvis.log", encoding="utf-8")],
    )
    # Suppress noisy libraries in console
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)


def create_pipeline(config):
    """Initialize all JARVIS components and build the pipeline."""
    from backend.app.agent.graph import build_jarvis_graph
    from backend.app.voice.audio import AudioPlayer, AudioRecorder
    from backend.app.voice.pipeline import VoicePipeline
    from backend.app.voice.stt import WhisperSTT
    from backend.app.voice.tts import create_tts_provider

    # Build the LangGraph agent
    console.print("  [cyan]►[/] Building agent brain (LangGraph + Ollama)...", end=" ")
    graph = build_jarvis_graph(
        model=config.ollama_model,
        base_url=config.ollama_url,
    )
    console.print("[green]✓[/]")

    # Local speech recognition works without a cloud TTS credential.
    console.print("  [cyan]►[/] Preparing local Whisper STT...", end=" ")
    stt = WhisperSTT(
        model_size=config.whisper_model_size,
        device=config.whisper_device,
        compute_type=config.whisper_compute_type,
    )
    console.print("[green]✓[/]")

    tts = None
    try:
        tts = create_tts_provider(
            config.tts_provider,
            api_key=config.elevenlabs_api_key,
            voice_id=config.elevenlabs_voice_id,
            kokoro_voice=config.kokoro_voice,
        )
        provider_name = "Kokoro (local)" if config.tts_provider == "kokoro" else "ElevenLabs"
        console.print(f"  [cyan]►[/] Preparing {provider_name} TTS... [green]✓[/]")
    except ValueError as exc:
        console.print(f"  [yellow]•[/] Voice replies are off: {exc}")

    pipeline = VoicePipeline(
        graph=graph,
        stt=stt,
        tts=tts,
        recorder=AudioRecorder(),
        player=AudioPlayer(),
    )

    return pipeline


def run_text_mode(pipeline) -> None:
    """Run JARVIS in text-only mode."""
    console.print(Panel(
        "[bold cyan]TEXT MODE[/] — Type your messages below.\n"
        "Commands: [bold]/voice[/] switch to voice │ [bold]/new[/] new conversation │ "
        "[bold]/quit[/] exit",
        title="💬 Text Mode",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = Prompt.ask("\n[bold green]You[/]")
        except (KeyboardInterrupt, EOFError):
            break

        if not user_input.strip():
            continue

        # Handle commands
        command = user_input.strip().lower()
        if command in ("/quit", "/exit", "/q"):
            break
        elif command == "/voice":
            if pipeline.voice_available:
                return run_voice_mode(pipeline)
            else:
                console.print("[red]Voice mode unavailable.[/] Set ELEVENLABS_API_KEY in .env")
                continue
        elif command == "/new":
            pipeline.new_conversation()
            console.print("[cyan]Started new conversation.[/]")
            continue
        elif command == "/help":
            console.print(
                "[cyan]/voice[/] — Switch to voice mode\n"
                "[cyan]/new[/]   — New conversation\n"
                "[cyan]/quit[/]  — Exit JARVIS"
            )
            continue

        # Process through agent
        console.print("[dim]JARVIS is thinking...[/]")
        start = time.monotonic()
        response = pipeline.process_text(user_input, mode="text", speak=False)
        elapsed = time.monotonic() - start

        # Display response with markdown rendering
        console.print()
        console.print(Panel(
            Markdown(response),
            title=f"🤖 JARVIS [dim]({elapsed:.1f}s)[/]",
            border_style="blue",
            padding=(1, 2),
        ))


def run_voice_mode(pipeline) -> None:
    """Run JARVIS in voice mode — speak and hear responses."""
    console.print(Panel(
        "[bold magenta]VOICE MODE[/] — Speak to JARVIS!\n"
        "Press [bold]Enter[/] to start recording (5 seconds).\n"
        "Commands: [bold]/text[/] switch to text │ [bold]/new[/] new conversation │ "
        "[bold]/quit[/] exit",
        title="🎤 Voice Mode",
        border_style="magenta",
    ))

    while True:
        try:
            user_input = Prompt.ask(
                "\n[bold magenta]Press Enter to speak, or type a command[/]",
                default="",
            )
        except (KeyboardInterrupt, EOFError):
            break

        command = user_input.strip().lower()
        if command in ("/quit", "/exit", "/q"):
            break
        elif command == "/text":
            return run_text_mode(pipeline)
        elif command == "/new":
            pipeline.new_conversation()
            console.print("[cyan]Started new conversation.[/]")
            continue
        elif command and not command.startswith("/"):
            # If they typed actual text in voice mode, still process it but speak
            console.print("[dim]JARVIS is thinking...[/]")
            response = pipeline.process_text(command, mode="voice", speak=True)
            console.print(f"\n[blue]🤖 JARVIS:[/] {response}")
            continue

        # Voice recording
        console.print("[bold red]🔴 Recording...[/] (speak now, 5 seconds)")
        try:
            response = pipeline.process_voice(record_seconds=5.0)
            console.print(f"\n[blue]🤖 JARVIS:[/] {response}")
        except Exception as exc:
            console.print(f"[red]Error:[/] {exc}")


def main() -> None:
    """Main entry point for the JARVIS CLI."""
    parser = argparse.ArgumentParser(description="JARVIS Voice Agent")
    parser.add_argument(
        "--mode", choices=["text", "voice"],
        help="Interaction mode (default: from .env or 'text')",
    )
    parser.add_argument("--model", help="Override Ollama model name")
    parser.add_argument("--server", action="store_true", help="Run as API server instead of CLI")
    args = parser.parse_args()

    setup_logging()

    # Show banner
    console.print(JARVIS_BANNER, style="bold cyan")
    console.print("[bold]Local AI Voice Assistant[/]", justify="center")
    console.print("[dim]Powered by LangGraph + Ollama + Whisper + ElevenLabs[/]\n", justify="center")

    # Load configuration
    from backend.app.config import JarvisConfig
    config = JarvisConfig.from_environment()

    # Override with CLI args
    if args.model:
        config = JarvisConfig(
            **{**config.__dict__, "ollama_model": args.model}  # type: ignore
        )

    # Run as API server
    if args.server:
        console.print("[cyan]Starting JARVIS API server...[/]")
        from backend.app.main import run_server
        run_server()
        return

    # Display config summary
    console.print(Panel(
        f"[cyan]Model:[/]  {config.ollama_model}\n"
        f"[cyan]Ollama:[/] {config.ollama_url}\n"
        f"[cyan]Voice:[/]  {'[green]Available[/]' if config.voice_enabled else '[red]Unavailable[/] (set ELEVENLABS_API_KEY)'}\n"
        f"[cyan]Whisper:[/] {config.whisper_model_size} on {config.whisper_device}",
        title="⚙️  Configuration",
        border_style="dim",
    ))

    # Initialize pipeline
    console.print("\n[bold]Initializing JARVIS...[/]")
    try:
        pipeline = create_pipeline(config)
    except Exception as exc:
        console.print(f"\n[bold red]Failed to initialize JARVIS:[/] {exc}")
        console.print("[yellow]Make sure Ollama is running: [bold]ollama serve[/][/]")
        sys.exit(1)

    console.print("[bold green]\n✅ JARVIS is ready![/]\n")

    # Handle graceful shutdown
    def signal_handler(sig, frame):
        console.print("\n[cyan]JARVIS shutting down... Goodbye! 👋[/]")
        pipeline.cleanup()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    # Determine starting mode
    start_mode = args.mode or config.mode
    if start_mode == "voice" and not pipeline.voice_available:
        console.print("[yellow]Voice mode requested but not available. Starting in text mode.[/]")
        start_mode = "text"

    # Run the interactive loop
    if start_mode == "voice":
        run_voice_mode(pipeline)
    else:
        run_text_mode(pipeline)

    # Cleanup
    pipeline.cleanup()
    console.print("\n[cyan]JARVIS offline. Goodbye! 👋[/]")


if __name__ == "__main__":
    main()
