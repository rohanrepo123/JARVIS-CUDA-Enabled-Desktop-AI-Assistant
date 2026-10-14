from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from backend.app.agent.prompts import (
    JARVIS_SYSTEM_PROMPT,
    JARVIS_TEXT_REMINDER,
    JARVIS_VOICE_REMINDER,
)
from backend.app.agent.state import JarvisState

logger = logging.getLogger(__name__)


# ── Node Functions ───────────────────────────────────────────────────
# Each function is a node in the LangGraph state machine.
# Nodes receive the current state and return state updates (partial dict).


def process_input(state: JarvisState) -> dict[str, Any]:
    """Process user input and prepare it for the agent.

    This is the entry node. It takes the current_input (text from typing
    or transcribed from voice) and adds it as a HumanMessage to the
    conversation history. Also injects or updates the system prompt with
    the correct mode instructions.
    """
    user_text = state.get("current_input", "")
    mode = state.get("mode", "text")

    if not user_text.strip():
        logger.warning("Empty input received, skipping")
        return {"response": "I didn't catch that. Could you try again?"}

    logger.info("Processing input [%s mode]: %s", mode, user_text[:80])

    # Build system message with mode-specific instructions
    mode_reminder = JARVIS_VOICE_REMINDER if mode == "voice" else JARVIS_TEXT_REMINDER
    system_content = f"{JARVIS_SYSTEM_PROMPT}\n\n{mode_reminder}"

    # Check if system message already exists in history
    messages = state.get("messages", [])
    has_system = any(isinstance(m, SystemMessage) for m in messages)

    new_messages = []
    if not has_system:
        new_messages.append(SystemMessage(content=system_content))
    else:
        # Update existing system message if mode changed (e.g. switched from text to voice)
        # We insert a brief mode-switch instruction as a SystemMessage
        last_system = next(
            (m for m in reversed(messages) if isinstance(m, SystemMessage)), None
        )
        if last_system and mode_reminder not in str(last_system.content):
            new_messages.append(SystemMessage(content=f"[Mode switched] {mode_reminder}"))

    new_messages.append(HumanMessage(content=user_text))

    return {"messages": new_messages}


def format_response(state: JarvisState) -> dict[str, Any]:
    """Extract the final response text from the last AI message.

    This runs after the agent (and any tool calls) have completed.
    It pulls the response text from the last AIMessage for TTS or display.
    Also sanitizes any raw JSON or tool-call artifacts that may have leaked
    through (e.g. when the LLM outputs raw tool JSON instead of natural text).
    """
    messages = state.get("messages", [])
    mode = state.get("mode", "text")

    # Find the last AI message
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            response_text = str(msg.content).strip()

            # Sanitize: if the response looks like raw JSON tool call, clean it up
            response_text = _sanitize_response(response_text, mode)

            logger.info("Response ready (%d chars)", len(response_text))
            return {"response": response_text}

    fallback = "I'm sorry, I couldn't generate a response. Please try again."
    return {"response": fallback}


def _sanitize_response(text: str, mode: str) -> str:
    """Clean up LLM output that may contain raw JSON tool calls or artifacts.

    Sometimes the LLM outputs raw JSON instead of natural language,
    especially when it tries to call tools that don't exist or when
    the response format is confused. This cleans that up.
    """
    stripped = text.strip()

    # Strip backticks and code fences — LLMs often wrap raw JSON in these
    cleaned = stripped
    # Remove code fences: ```json ... ``` or ``` ... ```
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    # Remove inline backticks
    cleaned = cleaned.strip("`").strip()

    # Check if the entire response is a JSON object (raw tool call leak)
    if cleaned.startswith("{") and cleaned.endswith("}"):
        try:
            data = json.loads(cleaned)
            # If it looks like a tool call, extract info and make it natural
            if "name" in data and ("parameters" in data or "arguments" in data):
                tool_name = data.get("name", "unknown")
                logger.warning("Sanitizing leaked tool call: %s", tool_name)
                # Make it conversational instead of showing raw JSON
                friendly_names = {
                    "get_current_time": "check the time",
                    "get_system_info": "check system status",
                    "search_web": "search the web",
                    "calculate": "do a calculation",
                    "open_website": "open a website",
                    "take_note": "save a note",
                    "set_reminder": "set a reminder",
                }
                action = friendly_names.get(tool_name, f"use {tool_name}")
                return f"Let me {action} for you. One moment please."
        except json.JSONDecodeError:
            pass

    # Check for JSON arrays too
    if cleaned.startswith("[") and cleaned.endswith("]"):
        try:
            json.loads(cleaned)
            logger.warning("Sanitizing leaked JSON array response")
            return "I encountered a formatting issue. Could you rephrase your question?"
        except json.JSONDecodeError:
            pass

    # In voice mode, strip markdown formatting for cleaner speech
    if mode == "voice":
        text = _strip_markdown_for_speech(text)

    return text


def _strip_markdown_for_speech(text: str) -> str:
    """Remove markdown formatting for cleaner text-to-speech output."""
    # Remove markdown headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove bold/italic markers
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", "", text)
    # Remove bullet points
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    # Remove numbered lists prefix
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Remove links, keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Clean up extra whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def should_use_tools(state: JarvisState) -> str:
    """Routing function: decide whether to call tools or go straight to response.

    Returns:
        'tools' if the last AI message contains tool calls.
        'respond' if no tools are needed.
    """
    messages = state.get("messages", [])
    if not messages:
        return "respond"

    last_message = messages[-1]

    # Check if the LLM wants to call tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        tool_names = [tc["name"] for tc in last_message.tool_calls]
        logger.info("Agent requesting tools: %s", tool_names)
        return "tools"

    return "respond"
