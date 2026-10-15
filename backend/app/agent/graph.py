from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta
from typing import Any
from uuid import uuid4

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from backend.app.agent.memory import create_memory_checkpointer
from backend.app.agent.nodes import format_response, process_input, should_use_tools
from backend.app.agent.ollama_llm import create_ollama_llm
from backend.app.agent.state import JarvisState
from backend.app.agent.tools import ALL_TOOLS

logger = logging.getLogger(__name__)
def _create_agent_node(llm_with_tools: BaseChatModel):
    """Create the agent node that calls the LLM with bound tools."""

    def agent(state: JarvisState) -> dict[str, Any]:
        """Call the LLM with the current conversation state and bound tools."""
        messages = state.get("messages", [])
        if messages and isinstance(messages[-1], HumanMessage):
            direct_call = _direct_productivity_tool_call(str(messages[-1].content))
            if direct_call:
                tool_name, arguments = direct_call
                if tool_name == "_clarify_alarm_time":
                    return {
                        "messages": [
                            AIMessage(
                                content=(
                                    '<response is_detail_insight_needed="false">'
                                    "<summary>What time should I set the alarm for? "
                                    "Please include AM or PM.</summary>"
                                    "<display_details></display_details></response>"
                                )
                            )
                        ]
                    }
                logger.info("Routing explicit productivity command to %s", tool_name)
                return {
                    "messages": [
                        AIMessage(
                            content="",
                            tool_calls=[
                                {
                                    "name": tool_name,
                                    "args": arguments,
                                    "id": uuid4().hex,
                                    "type": "tool_call",
                                }
                            ],
                        )
                    ]
                }
        logger.info("Agent thinking... (%d messages in context)", len(messages))
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return agent


def build_jarvis_graph(
    model: str = "llama3.2",
    base_url: str = "http://127.0.0.1:11434",
    temperature: float = 0.4,
) -> Any:
    """Build and compile the JARVIS LangGraph state machine.

    The graph follows this flow:
        START -> process_input -> agent -> should_use_tools?
            -> 'tools' -> tool_node -> agent (loop back)
            -> 'respond' -> format_response -> END

    Args:
        model: Ollama model name.
        base_url: Ollama server URL.
        temperature: LLM sampling temperature.

    Returns:
        A compiled LangGraph application with memory checkpointing.
    """
    logger.info("Building JARVIS graph: model=%s", model)

    # 1. Create the LLM and bind tools to it
    llm = create_ollama_llm(model=model, base_url=base_url, temperature=temperature)
    llm_with_tools = llm.bind_tools(ALL_TOOLS)

    # 2. Create the tool execution node (LangGraph prebuilt)
    tool_node = ToolNode(tools=ALL_TOOLS)

    # 3. Build the state graph
    workflow = StateGraph(JarvisState)

    # Add nodes
    workflow.add_node("process_input", process_input)
    workflow.add_node("agent", _create_agent_node(llm_with_tools))
    workflow.add_node("tools", tool_node)
    workflow.add_node("respond", format_response)

    # Define edges
    workflow.add_edge(START, "process_input")       # Entry point
    workflow.add_edge("process_input", "agent")     # Process input -> Call LLM
    workflow.add_conditional_edges(                   # Agent decides: tools or respond?
        "agent",
        should_use_tools,
        {
            "tools": "tools",
            "respond": "respond",
        },
    )
    workflow.add_edge("tools", "agent")              # After tools -> back to agent
    workflow.add_edge("respond", END)                # Response formatted -> done

    # 4. Compile with memory checkpointer for stateful conversations
    memory = create_memory_checkpointer()
    app = workflow.compile(checkpointer=memory)

    logger.info("JARVIS graph compiled successfully")
    return app


def _direct_productivity_tool_call(text: str) -> tuple[str, dict[str, Any]] | None:
    """Reliably route explicit alarm, reminder, and stopwatch commands."""
    normalized = " ".join(text.strip().lower().split())

    if re.search(r"\b(stop|pause|halt)\b.*\bstopwatch\b", normalized):
        return "stop_stopwatch", {}
    if re.search(r"\b(reset|clear)\b.*\bstopwatch\b", normalized):
        return "reset_stopwatch", {}
    if re.search(r"\b(start|begin|run|resume)\b.*\bstopwatch\b", normalized):
        return "start_stopwatch", {}
    if "stopwatch" in normalized and re.search(r"\b(check|status|time|elapsed|how long)\b", normalized):
        return "get_stopwatch", {}

    reminder_intent = re.search(r"\b(remind me|set (?:a )?reminder)\b", normalized)
    if reminder_intent:
        delay = re.search(
            r"\bin\s+(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
            normalized,
        )
        if delay:
            amount = int(delay.group(1))
            unit = delay.group(2)
            minutes = amount * (1 if unit.startswith(("minute", "min")) else
                                60 if unit.startswith(("hour", "hr")) else
                                1440 if unit.startswith("day") else 1 / 60)
            minutes = max(1, int(minutes))
            task = re.search(r"\bto\s+(.+?)(?=\s+in\s+\d+\s*(?:seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b|$)", normalized)
            message = task.group(1).strip(" .,!?") if task else "Reminder"
            return "set_reminder", {"message": message, "minutes": minutes}

    alarm_intent = re.search(
        r"\b(?:set|create|add|schedule)\s+(?:me\s+)?(?:an?\s+)?alarm\b"
        r"|\balarm\s+(?:for|at)\b|\bwake\s+me\s+up\b",
        normalized,
    )
    if alarm_intent:
        relative_time = re.search(
            r"\bin\s+(\d+)\s*(seconds?|secs?|minutes?|mins?|hours?|hrs?|days?)\b",
            normalized,
        )
        if relative_time:
            amount = int(relative_time.group(1))
            unit = relative_time.group(2)
            seconds = amount * (
                1 if unit.startswith(("second", "sec")) else
                60 if unit.startswith(("minute", "min")) else
                3600 if unit.startswith(("hour", "hr")) else 86400
            )
            moment = datetime.now().astimezone() + timedelta(seconds=seconds)
            return "set_alarm", {
                "alarm_time": moment.isoformat(timespec="seconds"),
                "label": "Alarm",
                "repeat": "none",
            }

        clock_time = re.search(
            r"\b(?P<hour>\d{1,2})(?::(?P<minute>[0-5]\d))?"
            r"\s*(?:o['’]?clock\s*)?(?P<period>a\.?m\.?|p\.?m\.?)"
            r"(?=\s|$|[,!?])",
            normalized,
        )
        if clock_time:
            hour = int(clock_time.group("hour"))
            minute = int(clock_time.group("minute") or 0)
            period = clock_time.group("period").replace(".", "")
            if not 1 <= hour <= 12:
                return None
            hour = hour % 12 + (12 if period == "pm" else 0)
        else:
            clock_time = re.search(
                r"\b(?P<hour>\d{1,2})(?::(?P<minute>[0-5]\d))?"
                r"\s+(?:in\s+the\s+)?(?P<period>morning|afternoon|evening|night|tonight|noon|midnight)\b",
                normalized,
            )
            if clock_time:
                hour = int(clock_time.group("hour"))
                minute = int(clock_time.group("minute") or 0)
                period = clock_time.group("period")
                if not 1 <= hour <= 12:
                    return None
                if period == "noon":
                    hour = 12
                elif period == "midnight":
                    hour = 0
                else:
                    hour = hour % 12 + (12 if period in {"afternoon", "evening", "night", "tonight"} else 0)
            else:
                clock_time = re.search(r"\b(?:at\s+)?([01]?\d|2[0-3]):([0-5]\d)\b", normalized)
                if not clock_time and "wake me up" in normalized:
                    # A wake-up request with a bare hour conventionally means AM.
                    clock_time = re.search(r"\b(?:at\s+)?(1[0-2]|0?[1-9])\b", normalized)
                    if clock_time:
                        hour, minute = int(clock_time.group(1)) % 12, 0
            if not clock_time:
                # Never guess whether an unqualified hour means AM or PM.
                return "_clarify_alarm_time", {}
            if "period" not in clock_time.groupdict() and clock_time.lastindex == 2:
                hour, minute = int(clock_time.group(1)), int(clock_time.group(2))
        recurrence = "daily" if re.search(r"\b(daily|every day)\b", normalized) else (
            "weekly" if re.search(r"\b(weekly|every week)\b", normalized) else "none"
        )
        if re.search(r"\b(today|tomorrow)\b", normalized):
            scheduled = datetime.now().astimezone() + timedelta(
                days=1 if "tomorrow" in normalized else 0
            )
            scheduled = scheduled.replace(hour=hour, minute=minute, second=0, microsecond=0)
            alarm_time = scheduled.isoformat(timespec="minutes")
        else:
            alarm_time = f"{hour:02d}:{minute:02d}"
        return "set_alarm", {
            "alarm_time": alarm_time,
            "label": "Alarm",
            "repeat": recurrence,
        }
    return None
