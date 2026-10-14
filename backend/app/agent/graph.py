from __future__ import annotations

import logging
from typing import Any

from langchain_core.language_models import BaseChatModel
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
