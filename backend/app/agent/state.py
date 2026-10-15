from __future__ import annotations

from typing import Annotated, Any

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages
from typing_extensions import TypedDict


class JarvisState(TypedDict):
    """State that flows through the JARVIS LangGraph agent.

    Attributes:
        messages: Conversation message history (auto-appended by LangGraph reducer).
        mode: Interaction mode — 'text' or 'voice'.
        current_input: The current user input text (from typing or transcription).
        response: The assistant's final response text for this turn.
        summary: Concise answer shown or spoken first in either mode.
        display_details: Optional additional on-screen details.
        is_detail_insight_needed: Whether optional details are useful for this turn.
        tool_results: Accumulated tool execution results for context.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    mode: str
    current_input: str
    response: str
    summary: str
    display_details: str
    is_detail_insight_needed: bool
    tool_results: list[dict[str, Any]]
