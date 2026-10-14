# JARVIS — CUDA-Enabled Desktop AI Assistant

> **Step 1 toward building JARVIS:** a local Windows desktop foundation with GPU-powered speech recognition, a local LLM connection, and a simple chat interface.

JARVIS is a personal AI assistant for Windows. It combines local voice transcription with a LangGraph-based assistant pipeline and Ollama. The included launcher opens JARVIS as a desktop app and keeps its setup files, AI-model cache, temporary files, and local data beside the project.

## What works

- Desktop chat window for talking with JARVIS
- Local Ollama model connection
- Voice transcription through faster-whisper
- NVIDIA CUDA support for Whisper on compatible GPUs
- Automatic CPU fallback if the GPU cannot process a transcription
- Conversation memory and built-in assistant tools
- Hold-to-talk voice input with an on-screen listening and transcription state
- Live streaming text replies while Ollama generates them
- Optional spoken replies for voice requests when ElevenLabs is configured
- Selectable local Kokoro or cloud ElevenLabs speech output
- Settings panel for voice replies and current model information

## How it works

```text
Microphone → Whisper speech-to-text → JARVIS / LangGraph → Ollama → response
```

## Requirements

- Windows 10 or 11
- Python 3.12, available through the `py` launcher
- [Ollama](https://ollama.com/) installed and running
- An Ollama model, for example:

  ```powershell
  ollama pull llama3.2
  ```

- For GPU transcription: an NVIDIA GPU with a current driver

## Run the desktop app

1. Clone or download this repository.
2. Copy `.env.example` to `.env` and set your preferred Ollama model and optional ElevenLabs values.
3. Double-click [Open-JARVIS.bat](Open-JARVIS.bat).

On the first launch, the script creates a private Python 3.12 environment in `../JEnv` and installs the project plus NVIDIA CUDA runtime packages. Later launches open the desktop window directly.

The launcher stores Hugging Face models, pip downloads, temporary files, and the JARVIS database inside the project folder. This prevents them from filling your C: drive when the project is on D:.

In **Settings**, choose **Kokoro · local CUDA** for on-device speech or **ElevenLabs · cloud** for online speech. Kokoro uses the CUDA 13.0 PyTorch wheel and downloads its model and voice assets on first use. ElevenLabs requires `ELEVENLABS_API_KEY` in `.env`.

## GPU voice setup

This project is configured for CUDA transcription:

```ini
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8_float16
```

The launcher installs the CUDA 12 cuBLAS, NVRTC, and cuDNN runtime libraries required by faster-whisper and CTranslate2. The application registers those DLL locations before it loads Whisper, fixing the common Windows error:

```text
Library cublas64_12.dll is not found or cannot be loaded
```

If CUDA is unavailable during loading or transcription, JARVIS automatically retries on CPU using `int8` mode.

## Command line mode

Activate the same environment and run one of these commands:

```powershell
..\JEnv\Scripts\Activate.ps1
python run.py
python run.py --mode voice
python run.py --server
```

## Project layout

```text
Jarvis_Desktop/
├── Open-JARVIS.bat       # Windows desktop launcher
├── desktop_app.py        # Tkinter desktop interface
├── run.py                # CLI and server entry point
├── requirements-lock.txt # Pinned Python dependencies
├── requirements-cuda.txt # NVIDIA CUDA runtime dependencies
├── requirements-kokoro-cuda.txt # Local Kokoro TTS and CUDA-enabled PyTorch
└── backend/app/
    ├── agent/            # LangGraph assistant and tools
    └── voice/            # Audio, Whisper STT, and TTS pipeline
```

## Roadmap

- [x] **Step 1 — Desktop foundation:** local chat, Ollama integration, CUDA Whisper, and a one-click Windows launcher.
- [x] **Step 2 — Voice-first interaction:** hold-to-talk recording, visible listening/transcribing/thinking states, voice transcripts in chat, streamed answers, and optional spoken replies.
- [ ] **Step 3 — Personal automation:** calendar, reminders, notes, and desktop actions.
- [ ] **Step 4 — JARVIS experience:** richer UI, settings, and persistent personal context.

## Security

Never commit `.env`. It can contain API keys. Use `.env.example` as the safe template for GitHub.

## Stack

- Python
- LangGraph and LangChain
- Ollama
- faster-whisper and CTranslate2
- NVIDIA CUDA runtime libraries
- PyTorch with CUDA 13.0 and Kokoro (local text-to-speech)
- ElevenLabs (optional cloud text-to-speech)
- Tkinter and FastAPI
