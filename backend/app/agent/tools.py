from __future__ import annotations

import json
import os
import platform
import sqlite3
import subprocess
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx
import psutil
from langchain_core.tools import tool


TMDB_BASE_URL = "https://api.themoviedb.org/3"
WEATHERSTACK_URL = "https://api.weatherstack.com/current"


def _tmdb_get(endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    """Call TMDB without exposing credentials in logs or user-facing errors."""
    api_key = os.getenv("TMDB_API_KEY", "").strip()
    if not api_key:
        return {"error": "TMDB is not configured. Add TMDB_API_KEY to your .env file."}

    query = {"api_key": api_key, **(params or {})}
    try:
        response = httpx.get(
            f"{TMDB_BASE_URL}/{endpoint.lstrip('/')}",
            params=query,
            timeout=12.0,
        )
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and data.get("success") is False:
            return {"error": data.get("status_message", "TMDB request failed.")}
        return data
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"TMDB request failed: {type(exc).__name__}."}


def _movie_card(movie: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": movie.get("id"),
        "title": movie.get("title") or movie.get("name"),
        "release_date": movie.get("release_date"),
        "rating": movie.get("vote_average"),
        "overview": (movie.get("overview") or "")[:500],
    }


def _movie_results(endpoint: str, params: dict[str, Any] | None = None, limit: int = 5) -> dict[str, Any]:
    data = _tmdb_get(endpoint, params)
    if "error" in data:
        return data
    return {"results": [_movie_card(item) for item in data.get("results", [])[:limit]]}


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
def movie_details(movie_title: str) -> dict[str, Any]:
    """Find a movie by title and return its release, runtime, genres, rating, and summary."""
    search = _tmdb_get("search/movie", {"query": movie_title, "include_adult": "false"})
    if "error" in search:
        return search
    matches = search.get("results", [])
    if not matches:
        return {"error": f"TMDB found no movie matching '{movie_title}'."}

    selected = matches[0]
    details = _tmdb_get(f"movie/{selected['id']}")
    if "error" in details:
        return details
    return {
        "movie": {
            **_movie_card(details),
            "runtime_minutes": details.get("runtime"),
            "genres": [genre.get("name") for genre in details.get("genres", [])],
            "original_language": details.get("original_language"),
            "tagline": details.get("tagline"),
        },
        "other_matches": [_movie_card(item) for item in matches[1:4]],
    }


@tool
def celebrity_details(name: str) -> dict[str, Any]:
    """Find a film celebrity and return a short profile and notable movie credits."""
    search = _tmdb_get("search/person", {"query": name, "include_adult": "false"})
    if "error" in search:
        return search
    matches = search.get("results", [])
    if not matches:
        return {"error": f"TMDB found no person matching '{name}'."}

    selected = matches[0]
    details = _tmdb_get(f"person/{selected['id']}", {"append_to_response": "movie_credits"})
    if "error" in details:
        return details
    credits = details.get("movie_credits", {}).get("cast", [])
    credits.sort(key=lambda item: item.get("popularity", 0), reverse=True)
    return {
        "name": details.get("name"),
        "known_for_department": details.get("known_for_department"),
        "biography": (details.get("biography") or "")[:1200],
        "birthday": details.get("birthday"),
        "place_of_birth": details.get("place_of_birth"),
        "notable_movies": [_movie_card(item) for item in credits[:5]],
    }


@tool
def similar_movies(movie_title: str) -> dict[str, Any]:
    """Recommend movies similar to a movie title supplied by the user."""
    search = _tmdb_get("search/movie", {"query": movie_title, "include_adult": "false"})
    if "error" in search:
        return search
    matches = search.get("results", [])
    if not matches:
        return {"error": f"TMDB found no movie matching '{movie_title}'."}
    movie = matches[0]
    recommendations = _movie_results(f"movie/{movie['id']}/recommendations", limit=5)
    return {"based_on": movie.get("title"), **recommendations}


@tool
def popular_movies(limit: int = 5) -> dict[str, Any]:
    """List currently popular movies. Limit results to between one and ten."""
    return _movie_results("movie/popular", limit=max(1, min(limit, 10)))


@tool
def trending_movies(time_window: str = "day", limit: int = 5) -> dict[str, Any]:
    """List trending movies for 'day' or 'week', with a small number of results."""
    window = time_window.strip().lower()
    if window not in {"day", "week"}:
        return {"error": "time_window must be 'day' or 'week'."}
    return _movie_results(
        f"trending/movie/{window}",
        limit=max(1, min(limit, 10)),
    )


@tool
def genre_based_movie_recommendation(genre: str, limit: int = 5) -> dict[str, Any]:
    """Find popular movies from a genre such as action, comedy, or science fiction."""
    genres = _tmdb_get("genre/movie/list")
    if "error" in genres:
        return genres
    wanted = genre.strip().casefold()
    genre_match = next(
        (item for item in genres.get("genres", []) if item.get("name", "").casefold() == wanted),
        None,
    )
    if not genre_match:
        return {
            "error": f"TMDB has no genre named '{genre}'.",
            "supported_genres": [item.get("name") for item in genres.get("genres", [])],
        }
    return {
        "genre": genre_match["name"],
        **_movie_results(
            "discover/movie",
            {"with_genres": genre_match["id"], "sort_by": "popularity.desc"},
            limit=max(1, min(limit, 10)),
        ),
    }


@tool
def get_weather_data(city: str) -> dict[str, Any]:
    """Get current metric weather conditions for a city using WeatherStack."""
    api_key = os.getenv("WEATHERSTACK_API_KEY", os.getenv("WEATHER_API", "")).strip()
    if not api_key:
        return {"error": "Weather is not configured. Add WEATHERSTACK_API_KEY to your .env file."}
    try:
        response = httpx.get(
            WEATHERSTACK_URL,
            params={"access_key": api_key, "query": city, "units": "m"},
            timeout=12.0,
        )
        response.raise_for_status()
        data = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        return {"error": f"Weather request failed: {type(exc).__name__}."}

    if "error" in data:
        return {"error": data["error"].get("info", "WeatherStack request failed.")}
    location = data.get("location", {})
    current = data.get("current", {})
    descriptions = current.get("weather_descriptions") or []
    return {
        "location": ", ".join(
            part for part in (location.get("name"), location.get("region"), location.get("country")) if part
        ),
        "local_time": location.get("localtime"),
        "temperature_c": current.get("temperature"),
        "feels_like_c": current.get("feelslike"),
        "conditions": descriptions[0] if descriptions else None,
        "humidity_percent": current.get("humidity"),
        "wind_speed_kmh": current.get("wind_speed"),
    }


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
    """Save a searchable note in JARVIS's local database.

    Args:
        title: A short title for the note.
        content: The note content to save.
    """
    try:
        note = _productivity_store().save_note(title, content)
        return f"Saved note {note['id']}: {note['title']}. You can find it later by searching your notes."
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't save that note: {exc}"


@tool
def search_notes(query: str = "", limit: int = 10) -> list[dict[str, Any]] | str:
    """Search saved JARVIS notes, or list recent notes when query is empty."""
    try:
        notes = _productivity_store().search_notes(query, limit)
        return notes or "No matching saved notes."
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't search notes: {exc}"


@tool
def remember_preference(key: str, value: str) -> str:
    """Save a user preference for future JARVIS conversations."""
    try:
        _productivity_store().save_preference(key, value)
        return f"I'll remember {key.strip()}: {value.strip()}."
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't save that preference: {exc}"


@tool
def get_user_preferences() -> list[dict[str, Any]] | str:
    """Retrieve preferences the user has asked JARVIS to remember."""
    try:
        preferences = _productivity_store().list_preferences()
        return preferences or "No saved preferences yet."
    except (OSError, sqlite3.Error) as exc:
        return f"I couldn't retrieve preferences: {exc}"


@tool
def save_conversation_summary(title: str, summary: str) -> str:
    """Save a concise, user-approved summary of a conversation in local memory."""
    try:
        item = _productivity_store().save_conversation_summary(title, summary)
        return f"Saved conversation summary {item['id']}: {item['title']}."
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't save that conversation summary: {exc}"


@tool
def search_conversation_summaries(query: str = "", limit: int = 10) -> list[dict[str, Any]] | str:
    """Search locally saved conversation summaries, or list recent summaries."""
    try:
        items = _productivity_store().search_conversation_summaries(query, limit)
        return items or "No matching saved conversation summaries."
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't search conversation summaries: {exc}"


@tool
def set_reminder(message: str, minutes: int = 1) -> str:
    """Set a persistent reminder for a number of minutes from now.

    Args:
        message: The reminder message.
        minutes: How many minutes from now (default 1).
    """
    try:
        reminder = _productivity_store().add_timed_reminder(message, minutes)
        when = datetime.fromisoformat(reminder["due_at"]).strftime("%I:%M %p")
        return f"Reminder {reminder['id']} set for {when}: {reminder['message']}"
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't set that reminder: {exc}"


@tool
def schedule_reminder(message: str, due_at: str, repeat: str = "none") -> str:
    """Schedule a persistent reminder at a local ISO datetime, e.g. 2026-10-20T09:30."""
    try:
        moment = datetime.fromisoformat(due_at)
        if moment.tzinfo is None:
            moment = moment.astimezone()
        reminder = _productivity_store().add_reminder(message, moment, repeat)
        return f"Reminder {reminder['id']} set for {reminder['due_at']}: {reminder['message']} ({repeat})."
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't schedule that reminder: {exc}"


@tool
def set_alarm(alarm_time: str, label: str = "Alarm", repeat: str = "none") -> str:
    """Set a persistent JARVIS alarm for HH:MM (24-hour local time) or ISO datetime."""
    try:
        alarm = _productivity_store().add_alarm(alarm_time, label, repeat)
        return f"JARVIS alarm {alarm['id']} is set for {alarm['due_at']}: {alarm['message']} ({repeat})."
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't set that alarm: {exc}"


@tool
def list_reminders() -> list[dict[str, Any]] | str:
    """List active reminders and alarms with their scheduled times."""
    try:
        reminders = _productivity_store().list_reminders()
        return reminders or "There are no active reminders or alarms."
    except (OSError, sqlite3.Error) as exc:
        return f"I couldn't list reminders: {exc}"


@tool
def cancel_reminder(reminder_id: int) -> str:
    """Cancel an active reminder or alarm by its ID."""
    try:
        if _productivity_store().cancel_reminder(reminder_id):
            return f"Cancelled reminder or alarm {reminder_id}."
        return f"No active reminder or alarm with ID {reminder_id} was found."
    except (OSError, ValueError, sqlite3.Error) as exc:
        return f"I couldn't cancel that reminder: {exc}"


@tool
def start_stopwatch() -> str:
    """Start or resume JARVIS's persistent stopwatch."""
    return _stopwatch_message("Started", "start")


@tool
def stop_stopwatch() -> str:
    """Pause JARVIS's stopwatch and report elapsed time."""
    return _stopwatch_message("Stopped", "stop")


@tool
def reset_stopwatch() -> str:
    """Reset JARVIS's stopwatch to zero."""
    return _stopwatch_message("Reset", "reset")


@tool
def get_stopwatch() -> str:
    """Report whether JARVIS's stopwatch is running and its elapsed time."""
    try:
        state = _productivity_store().stopwatch()
        status = "running" if state["running"] else "paused"
        return f"Stopwatch is {status} at {state['display']}."
    except (OSError, sqlite3.Error) as exc:
        return f"I couldn't read the stopwatch: {exc}"


@tool
def open_windows_clock(section: str = "") -> str:
    """Open the Windows Clock app. Optionally name alarms, timer, or stopwatch."""
    if platform.system() != "Windows":
        return "Windows Clock is available only on Windows."
    section = section.strip().lower()
    if section and section not in {"alarm", "alarms", "timer", "stopwatch"}:
        return "Choose alarms, timer, or stopwatch."
    try:
        os.startfile("ms-clock:")  # type: ignore[attr-defined]
        view = f". Select {section} in Clock" if section else ""
        return f"Opened Windows Clock{view}. JARVIS alarms and stopwatch are managed separately in the assistant."
    except (OSError, sqlite3.Error) as exc:
        return f"I couldn't open Windows Clock: {exc}"


@tool
def open_application(application: str) -> str:
    """Open a safe built-in Windows app: notepad, calculator, explorer, or clock."""
    if platform.system() != "Windows":
        return "Opening built-in apps is available only on Windows."
    commands = {
        "notepad": ["notepad.exe"],
        "calculator": ["calc.exe"],
        "explorer": ["explorer.exe"],
    }
    name = application.strip().lower()
    if name in {"clock", "windows clock", "alarms"}:
        return open_windows_clock.invoke({"section": ""})
    command = commands.get(name)
    if not command:
        return "I can open Notepad, Calculator, File Explorer, or Windows Clock."
    try:
        subprocess.Popen(command, close_fds=True)
        return f"Opened {name.title()}."
    except (OSError, sqlite3.Error) as exc:
        return f"I couldn't open {name}: {exc}"


@tool
def open_local_path(path: str) -> str:
    """Open an existing local file or folder, but never launch scripts or programs."""
    if platform.system() != "Windows":
        return "Opening a local path this way is available only on Windows."
    target = Path(path).expanduser()
    try:
        target = target.resolve(strict=True)
        if target.suffix.lower() in {".exe", ".com", ".bat", ".cmd", ".ps1", ".msi", ".lnk", ".vbs", ".js"}:
            return "For safety, JARVIS won't launch executable files or scripts."
        if not (target.is_file() or target.is_dir()):
            return "That path is not a file or folder."
        os.startfile(str(target))  # type: ignore[attr-defined]
        return f"Opened {target.name}."
    except (OSError, RuntimeError) as exc:
        return f"I couldn't open that path: {exc}"


def _productivity_store():
    from backend.app.productivity import ProductivityStore

    return ProductivityStore()


def _stopwatch_message(action: str, operation: str) -> str:
    try:
        store = _productivity_store()
        if operation == "start":
            state = store.start_stopwatch()
        elif operation == "stop":
            state = store.stop_stopwatch()
        else:
            state = store.reset_stopwatch()
        return f"{action} stopwatch. Elapsed time: {state['display']}."
    except (OSError, sqlite3.Error) as exc:
        return f"I couldn't update the stopwatch: {exc}"


# ── Tool Collection ──────────────────────────────────────────────────

ALL_TOOLS = [
    get_current_time,
    get_system_info,
    search_web,
    movie_details,
    celebrity_details,
    similar_movies,
    popular_movies,
    trending_movies,
    genre_based_movie_recommendation,
    get_weather_data,
    open_website,
    calculate,
    take_note,
    set_reminder,
    search_notes,
    remember_preference,
    get_user_preferences,
    save_conversation_summary,
    search_conversation_summaries,
    schedule_reminder,
    set_alarm,
    list_reminders,
    cancel_reminder,
    start_stopwatch,
    stop_stopwatch,
    reset_stopwatch,
    get_stopwatch,
    open_windows_clock,
    open_application,
    open_local_path,
]
