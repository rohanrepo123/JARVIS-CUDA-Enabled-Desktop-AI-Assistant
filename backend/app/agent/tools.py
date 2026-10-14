from __future__ import annotations

import json
import platform
import webbrowser
from datetime import datetime
from typing import Any
from urllib.parse import quote_plus

import httpx
import psutil
from langchain_core.tools import tool


# ── Read-Only Tools (R1 — No approval needed) ────────────────────────

@tool
def get_current_time() -> str:
    """Get the current local date, time, and timezone on this computer."""
    now = datetime.now().astimezone()
    return now.strftime("%A, %B %d, %Y at %I:%M %p (%Z)")


@tool
def get_system_info() -> dict[str, Any]:
    """Get system hardware information: CPU usage, memory, disk, OS details."""
    memory = psutil.virtual_memory()
    try:
        disk = psutil.disk_usage("/" if platform.system() != "Windows" else "C:\\")
        disk_info = {
            "disk_used_percent": disk.percent,
            "disk_total_gb": round(disk.total / (1024 ** 3), 1),
            "disk_free_gb": round(disk.free / (1024 ** 3), 1),
        }
    except OSError:
        disk_info = {"disk_used_percent": None, "disk_total_gb": None, "disk_free_gb": None}

    return {
        "os": f"{platform.system()} {platform.release()}",
        "processor": platform.processor()[:100] or "Unknown",
        "cpu_usage_percent": psutil.cpu_percent(interval=0.5),
        "cpu_cores": psutil.cpu_count(),
        "memory_used_percent": memory.percent,
        "memory_total_gb": round(memory.total / (1024 ** 3), 1),
        "memory_available_gb": round(memory.available / (1024 ** 3), 1),
        **disk_info,
    }


@tool
def search_web(query: str) -> str:
    """Search the web for current information using DuckDuckGo.

    Use this when the user asks about current events, facts you're unsure about,
    or anything that requires up-to-date information.

    Args:
        query: The search query string (e.g., 'latest news about AI').
    """
    try:
        # Use DuckDuckGo Instant Answer API (no API key needed)
        url = "https://api.duckduckgo.com/"
        params = {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        response = httpx.get(url, params=params, timeout=10, follow_redirects=True)
        response.raise_for_status()
        data = response.json()

        results: list[str] = []

        # Abstract (main answer)
        if data.get("AbstractText"):
            results.append(f"Summary: {data['AbstractText']}")
            if data.get("AbstractSource"):
                results.append(f"Source: {data['AbstractSource']}")

        # Answer (direct answer)
        if data.get("Answer"):
            results.append(f"Answer: {data['Answer']}")

        # Related topics
        for topic in data.get("RelatedTopics", [])[:3]:
            if isinstance(topic, dict) and topic.get("Text"):
                results.append(f"- {topic['Text'][:200]}")

        if results:
            return "\n".join(results)

        # Fallback: return a helpful message with search link
        search_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
        return (
            f"I couldn't find a direct answer for '{query}'. "
            f"You can search here: {search_url}"
        )
    except Exception as exc:
        return f"Web search failed: {exc}. Try asking in a different way."


@tool
def open_website(url: str) -> str:
    """Open a website URL in the user's default web browser.

    Args:
        url: The full URL to open (e.g., 'https://google.com').
    """
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened {url} in your browser."


@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Args:
        expression: A math expression like '2 + 2', 'sqrt(144)', '15 * 27 + 300'.
    """
    import math

    # Clean up natural language math
    expr = expression.replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/")
    expr = expr.replace(" of ", "*")
    if "%" in expr:
        expr = expr.replace("%", "/100")

    allowed = {
        "__builtins__": {},
        "abs": abs, "round": round, "min": min, "max": max, "sum": sum,
        "pow": pow, "int": int, "float": float,
        "sqrt": math.sqrt, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "pi": math.pi, "e": math.e,
    }
    try:
        result = eval(expr, allowed)  # noqa: S307 — sandboxed with restricted builtins
        return f"{expression} = {result}"
    except Exception as exc:
        return f"Could not calculate '{expression}': {exc}"


@tool
def take_note(title: str, content: str) -> str:
    """Save a quick note to the JARVIS workspace.

    Args:
        title: A short title for the note.
        content: The note content to save.
    """
    import os
    from pathlib import Path

    workspace = Path(os.getenv("JARVIS_WORKSPACE", str(Path.home() / "JARVIS-Workspace")))
    notes_dir = workspace / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_title = "".join(c if c.isalnum() or c in " -_" else "" for c in title)[:50].strip()
    filename = f"{timestamp}_{safe_title or 'note'}.md"
    filepath = notes_dir / filename

    filepath.write_text(f"# {title}\n\n{content}\n", encoding="utf-8")
    return f"Note saved: {filepath}"


@tool
def set_reminder(message: str, minutes: int = 1) -> str:
    """Set a reminder that will be shown after the specified number of minutes.

    Args:
        message: The reminder message.
        minutes: How many minutes from now (default 1).
    """
    import threading
    from datetime import timedelta

    def _remind():
        print(f"\n\U0001f514 REMINDER: {message}")

    timer = threading.Timer(minutes * 60, _remind)
    timer.daemon = True
    timer.start()

    remind_time = datetime.now().astimezone() + timedelta(minutes=minutes)
    return f"Reminder set for {remind_time.strftime('%I:%M %p')}: {message}"


# ── Tool Collection ──────────────────────────────────────────────────

ALL_TOOLS = [
    get_current_time,
    get_system_info,
    search_web,
    open_website,
    calculate,
    take_note,
    set_reminder,
]
