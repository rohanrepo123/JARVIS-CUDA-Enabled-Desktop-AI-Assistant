from __future__ import annotations

import json
import logging
import re
import sqlite3
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage

from backend.app.agent.prompts import (
    JARVIS_SYSTEM_PROMPT,
    JARVIS_TEXT_REMINDER,
    JARVIS_VOICE_REMINDER,
)
from backend.app.agent.response_format import AssistantResponseStream
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
    try:
        from backend.app.productivity import ProductivityStore

        preferences = ProductivityStore().list_preferences()
        if preferences:
            saved_context = "\n".join(
                f"- {item['key']}: {item['value']}" for item in preferences[:30]
            )
            system_content += f"\n\nSaved user preferences (private local memory):\n{saved_context}"
    except (OSError, ValueError, sqlite3.Error) as exc:
        logger.warning("Could not load saved preferences: %s", exc)

    # Refresh the system prompt each turn so existing checkpointed conversations
    # also receive the latest shared response schema and mode guidance.
    messages = state.get("messages", [])
    new_messages = [
        RemoveMessage(id=message.id)
        for message in messages
        if isinstance(message, SystemMessage) and message.id
    ]
    new_messages.append(SystemMessage(content=system_content))
    new_messages.append(HumanMessage(content=user_text))

    return {"messages": new_messages}


def format_response(state: JarvisState) -> dict[str, Any]:
    """Extract and normalize the summary/details schema from the final AI message.

    This runs after the agent (and any tool calls) have completed.
    It removes the internal envelope from the user-visible response and replaces
    the raw model message in memory with the rendered answer.
    """
    messages = state.get("messages", [])
    mode = state.get("mode", "text")

    # Find the last AI message
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and msg.content:
            raw_response = str(msg.content).strip()

            parsed = AssistantResponseStream()
            parsed.feed(raw_response)
            parsed.finish()

            if parsed.valid and parsed.summary_complete:
                summary = parsed.summary
                details = parsed.details if parsed.is_detail_insight_needed else ""
                details_needed = bool(parsed.is_detail_insight_needed and details)
            else:
                # A model may occasionally ignore the requested format. Preserve its
                # answer and keep the UI usable without showing internal schema tags.
                plain = _sanitize_response(raw_response, "text")
                summary = plain.strip()
                details = ""
                details_needed = False

            if mode == "voice":
                if not (parsed.valid and parsed.summary_complete):
                    summary = _summarize_for_speech(summary)
                summary = _strip_markdown_for_speech(summary).strip()

            latest_user = next(
                (message for message in reversed(messages) if isinstance(message, HumanMessage)),
                None,
            )
            if latest_user is not None:
                last_user_index = max(
                    index
                    for index, message in enumerate(messages)
                    if isinstance(message, HumanMessage)
                )
                current_turn = messages[last_user_index + 1:]
                corrected_summary = _verify_productivity_claim(
                    str(latest_user.content), current_turn, summary
                )
                if corrected_summary != summary:
                    summary = corrected_summary
                    details = ""
                    details_needed = False
            rendered = summary
            if details_needed:
                rendered = f"{summary}\n\nDetails\n{details}"

            updates: dict[str, Any] = {
                "response": rendered,
                "summary": summary,
                "display_details": details,
                "is_detail_insight_needed": details_needed,
            }
            clean_message = AIMessage(content=rendered)
            if msg.id:
                updates["messages"] = [RemoveMessage(id=msg.id), clean_message]
            else:
                updates["messages"] = [clean_message]
            logger.info("Response ready (%d chars; details=%s)", len(rendered), details_needed)
            return updates

    fallback = "I'm sorry, I couldn't generate a response. Please try again."
    return {"response": fallback}


def _verify_productivity_claim(
    user_text: str, current_turn: list[Any], response: str
) -> str:
    """Reject success claims about alarms/reminders unless a tool confirmed them."""
    request = user_text.lower()
    answer = response.lower()
    if not re.search(r"\b(alarm|reminders?)\b|\bremind\s+me\b|\bwake\s+me\s+up\b", request):
        return response
    if not re.search(
        r"\b(set|create|add|schedule|remind me|cancel)\b"
        r"|\balarm\s+(?:for|at)\b|\bwake\s+me\s+up\b",
        request,
    ):
        return response
    claims_success = bool(
        re.search(
            r"\b(?:your|the|an?)\s+(?:alarm|reminder)\b.{0,80}\b(?:is set|has been set|was set|set for|set at|created|scheduled|added|cancelled|canceled)\b"
            r"|\b(?:set|created|scheduled|added|cancelled|canceled)\b.{0,80}\b(?:alarm|reminder)\b"
            r"|\b(?:alarm|reminder)\b.{0,80}\b(?:set for|set at|created|scheduled|added|cancelled|canceled)\b",
            answer,
        )
    )
    if not claims_success:
        return response

    allowed_tools = {"set_alarm", "set_reminder", "schedule_reminder", "cancel_reminder"}
    tool_messages = [
        message
        for message in current_turn
        if isinstance(message, ToolMessage) and message.name in allowed_tools
    ]
    for message in tool_messages:
        result = str(message.content).lower()
        if not re.search(r"\b(couldn't|could not|failed|error|must be|not found)\b", result):
            return response
    if tool_messages:
        outcome = str(tool_messages[-1].content).strip()
        return f"I couldn't complete that request: {outcome}"
    return "I couldn't confirm that the alarm or reminder was saved, so I won't say it was set. Please try again with the time or delay."


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
    # If the model emitted a partial or malformed response envelope, keep its
    # useful prose while hiding implementation-only tags from the user.
    cleaned = AssistantResponseStream._clean_markup(cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()

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

    # Keep the sanitized value. Returning the original text here reintroduced
    # raw schema tags precisely when parsing had fallen back from a malformed
    # envelope.
    text = cleaned

    # In voice mode, strip markdown formatting for cleaner speech
    if mode == "voice":
        text = _strip_markdown_for_speech(text)

    return text


def _summarize_for_speech(text: str, max_sentences: int = 3, max_chars: int = 320) -> str:
    """Shorten unstructured fallback output for a voice turn."""
    clean = _strip_markdown_for_speech(text)
    sentences = re.split(r"(?<=[.!?])\s+", clean)
    summary = " ".join(sentence for sentence in sentences[:max_sentences] if sentence)
    if len(summary) > max_chars:
        summary = summary[:max_chars].rsplit(" ", 1)[0].rstrip(" ,;:") + "."
    return summary or "I couldn't prepare a spoken answer. Please try again."


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
