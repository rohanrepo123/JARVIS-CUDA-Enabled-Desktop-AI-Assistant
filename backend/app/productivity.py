from __future__ import annotations

import os
import sqlite3
import threading
import time
import logging
from contextlib import contextmanager
from collections.abc import Callable, Iterator
from datetime import datetime, time as day_time, timedelta
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ProductivityStore:
    """Persistent local storage for JARVIS reminders, notes, preferences, and timer."""

    def __init__(self, database_path: str | Path | None = None) -> None:
        if database_path:
            self.database_path = Path(database_path)
        else:
            configured = os.getenv("JARVIS_DATABASE_PATH", "").strip()
            if configured:
                self.database_path = Path(configured)
            else:
                local_data = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
                self.database_path = local_data / "JARVIS" / "jarvis.db"
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 10000")
        return connection

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self._session() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    message TEXT NOT NULL,
                    due_at REAL NOT NULL,
                    recurrence TEXT NOT NULL DEFAULT 'none',
                    kind TEXT NOT NULL DEFAULT 'reminder',
                    active INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_reminders_due
                    ON reminders(active, due_at);
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS preferences (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    title TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS stopwatch (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    started_at REAL,
                    elapsed_seconds REAL NOT NULL DEFAULT 0
                );
                INSERT OR IGNORE INTO stopwatch (id, started_at, elapsed_seconds)
                    VALUES (1, NULL, 0);
                """
            )

    @staticmethod
    def _local_iso(timestamp: float) -> str:
        return datetime.fromtimestamp(timestamp).astimezone().isoformat(timespec="minutes")

    def add_reminder(
        self,
        message: str,
        due_at: datetime,
        recurrence: str = "none",
        kind: str = "reminder",
    ) -> dict[str, Any]:
        recurrence = recurrence.strip().lower()
        if recurrence not in {"none", "daily", "weekly"}:
            raise ValueError("Repeat must be none, daily, or weekly.")
        if kind not in {"reminder", "alarm"}:
            raise ValueError("Unsupported reminder type.")
        if due_at.tzinfo is None:
            due_at = due_at.astimezone()
        message = message.strip()[:500]
        if not message:
            raise ValueError("A reminder message is required.")
        due_epoch = due_at.timestamp()
        with self._session() as connection:
            cursor = connection.execute(
                "INSERT INTO reminders(message, due_at, recurrence, kind, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (message, due_epoch, recurrence, kind, time.time()),
            )
            reminder_id = cursor.lastrowid
        return {
            "id": reminder_id,
            "message": message,
            "due_at": self._local_iso(due_epoch),
            "recurrence": recurrence,
            "kind": kind,
        }

    def add_timed_reminder(
        self, message: str, minutes: int, recurrence: str = "none"
    ) -> dict[str, Any]:
        minutes = max(1, min(int(minutes), 525600))
        return self.add_reminder(message, datetime.now().astimezone() + timedelta(minutes=minutes), recurrence)

    def add_alarm(self, when: str, label: str = "Alarm", recurrence: str = "none") -> dict[str, Any]:
        value = when.strip()
        now = datetime.now().astimezone()
        try:
            due_at = datetime.fromisoformat(value)
            if due_at.tzinfo is None:
                due_at = due_at.astimezone()
        except ValueError:
            try:
                alarm_time = day_time.fromisoformat(value)
            except ValueError as exc:
                raise ValueError("Use a 24-hour time such as 07:30 or an ISO date and time.") from exc
            due_at = now.replace(
                hour=alarm_time.hour,
                minute=alarm_time.minute,
                second=0,
                microsecond=0,
            )
            if due_at <= now:
                due_at += timedelta(days=1)
        if due_at <= now:
            raise ValueError("Alarm time must be in the future.")
        return self.add_reminder(label or "Alarm", due_at, recurrence, kind="alarm")

    def list_reminders(self, include_inactive: bool = False) -> list[dict[str, Any]]:
        query = "SELECT id, message, due_at, recurrence, kind, active FROM reminders"
        if not include_inactive:
            query += " WHERE active = 1"
        query += " ORDER BY due_at LIMIT 100"
        with self._session() as connection:
            rows = connection.execute(query).fetchall()
        return [
            {
                "id": row["id"],
                "message": row["message"],
                "due_at": self._local_iso(row["due_at"]),
                "recurrence": row["recurrence"],
                "kind": row["kind"],
                "active": bool(row["active"]),
            }
            for row in rows
        ]

    def cancel_reminder(self, reminder_id: int) -> bool:
        with self._session() as connection:
            cursor = connection.execute(
                "UPDATE reminders SET active = 0 WHERE id = ? AND active = 1",
                (int(reminder_id),),
            )
            return cursor.rowcount > 0

    def take_due_reminders(self, now: float | None = None) -> list[dict[str, Any]]:
        now = time.time() if now is None else now
        due_events: list[dict[str, Any]] = []
        with self._session() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT id, message, due_at, recurrence, kind FROM reminders "
                "WHERE active = 1 AND due_at <= ? ORDER BY due_at",
                (now,),
            ).fetchall()
            for row in rows:
                due_events.append(
                    {
                        "id": row["id"],
                        "message": row["message"],
                        "due_at": self._local_iso(row["due_at"]),
                        "recurrence": row["recurrence"],
                        "kind": row["kind"],
                    }
                )
                recurrence = row["recurrence"]
                if recurrence == "none":
                    connection.execute("UPDATE reminders SET active = 0 WHERE id = ?", (row["id"],))
                    continue
                next_due = datetime.fromtimestamp(row["due_at"]).astimezone()
                step = timedelta(days=1 if recurrence == "daily" else 7)
                while next_due.timestamp() <= now:
                    next_due += step
                connection.execute(
                    "UPDATE reminders SET due_at = ? WHERE id = ?",
                    (next_due.timestamp(), row["id"]),
                )
        return due_events

    def save_note(self, title: str, content: str) -> dict[str, Any]:
        title = title.strip()[:200]
        content = content.strip()[:20000]
        if not title or not content:
            raise ValueError("A note needs both a title and content.")
        now = time.time()
        with self._session() as connection:
            cursor = connection.execute(
                "INSERT INTO notes(title, content, created_at, updated_at) VALUES (?, ?, ?, ?)",
                (title, content, now, now),
            )
            note_id = cursor.lastrowid
        return {"id": note_id, "title": title, "created_at": self._local_iso(now)}

    def search_notes(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        query = query.strip()[:200]
        limit = max(1, min(int(limit), 50))
        with self._session() as connection:
            if query:
                escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                rows = connection.execute(
                    "SELECT id, title, content, created_at, updated_at FROM notes "
                    "WHERE title LIKE ? ESCAPE '\\' OR content LIKE ? ESCAPE '\\' "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (f"%{escaped}%", f"%{escaped}%", limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, title, content, created_at, updated_at FROM notes "
                    "ORDER BY updated_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "content": row["content"],
                "created_at": self._local_iso(row["created_at"]),
                "updated_at": self._local_iso(row["updated_at"]),
            }
            for row in rows
        ]

    def delete_note(self, note_id: int) -> bool:
        with self._session() as connection:
            cursor = connection.execute("DELETE FROM notes WHERE id = ?", (int(note_id),))
            return cursor.rowcount > 0

    def save_preference(self, key: str, value: str) -> None:
        key = key.strip()[:100]
        value = value.strip()[:1000]
        if not key or not value:
            raise ValueError("A preference needs a key and value.")
        with self._session() as connection:
            connection.execute(
                "INSERT INTO preferences(key, value, updated_at) VALUES (?, ?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at",
                (key, value, time.time()),
            )

    def list_preferences(self) -> list[dict[str, Any]]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT key, value, updated_at FROM preferences ORDER BY key COLLATE NOCASE"
            ).fetchall()
        return [
            {"key": row["key"], "value": row["value"], "updated_at": self._local_iso(row["updated_at"])}
            for row in rows
        ]

    def delete_preference(self, key: str) -> bool:
        with self._session() as connection:
            cursor = connection.execute("DELETE FROM preferences WHERE key = ?", (key,))
            return cursor.rowcount > 0

    def save_conversation_summary(self, title: str, summary: str) -> dict[str, Any]:
        title = title.strip()[:200]
        summary = summary.strip()[:20000]
        if not title or not summary:
            raise ValueError("A conversation summary needs a title and summary text.")
        created_at = time.time()
        with self._session() as connection:
            cursor = connection.execute(
                "INSERT INTO conversation_summaries(title, summary, created_at) VALUES (?, ?, ?)",
                (title, summary, created_at),
            )
            summary_id = cursor.lastrowid
        return {"id": summary_id, "title": title, "created_at": self._local_iso(created_at)}

    def search_conversation_summaries(self, query: str = "", limit: int = 10) -> list[dict[str, Any]]:
        query = query.strip()[:200]
        limit = max(1, min(int(limit), 50))
        with self._session() as connection:
            if query:
                escaped = query.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                rows = connection.execute(
                    "SELECT id, title, summary, created_at FROM conversation_summaries "
                    "WHERE title LIKE ? ESCAPE '\\' OR summary LIKE ? ESCAPE '\\' "
                    "ORDER BY created_at DESC LIMIT ?",
                    (f"%{escaped}%", f"%{escaped}%", limit),
                ).fetchall()
            else:
                rows = connection.execute(
                    "SELECT id, title, summary, created_at FROM conversation_summaries "
                    "ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "id": row["id"],
                "title": row["title"],
                "summary": row["summary"],
                "created_at": self._local_iso(row["created_at"]),
            }
            for row in rows
        ]

    def delete_conversation_summary(self, summary_id: int) -> bool:
        with self._session() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_summaries WHERE id = ?", (int(summary_id),)
            )
            return cursor.rowcount > 0

    def clear_saved_memory(self) -> None:
        with self._session() as connection:
            connection.execute("DELETE FROM notes")
            connection.execute("DELETE FROM preferences")
            connection.execute("DELETE FROM conversation_summaries")

    def stopwatch(self) -> dict[str, Any]:
        with self._session() as connection:
            row = connection.execute(
                "SELECT started_at, elapsed_seconds FROM stopwatch WHERE id = 1"
            ).fetchone()
        elapsed = float(row["elapsed_seconds"])
        running = row["started_at"] is not None
        if running:
            elapsed += max(0.0, time.time() - float(row["started_at"]))
        return {"running": running, "elapsed_seconds": round(elapsed, 2), "display": self._format_duration(elapsed)}

    def start_stopwatch(self) -> dict[str, Any]:
        with self._session() as connection:
            connection.execute(
                "UPDATE stopwatch SET started_at = COALESCE(started_at, ?) WHERE id = 1",
                (time.time(),),
            )
        return self.stopwatch()

    def stop_stopwatch(self) -> dict[str, Any]:
        now = time.time()
        with self._session() as connection:
            row = connection.execute(
                "SELECT started_at, elapsed_seconds FROM stopwatch WHERE id = 1"
            ).fetchone()
            if row["started_at"] is not None:
                elapsed = float(row["elapsed_seconds"]) + max(0.0, now - float(row["started_at"]))
                connection.execute(
                    "UPDATE stopwatch SET started_at = NULL, elapsed_seconds = ? WHERE id = 1",
                    (elapsed,),
                )
        return self.stopwatch()

    def reset_stopwatch(self) -> dict[str, Any]:
        with self._session() as connection:
            connection.execute(
                "UPDATE stopwatch SET started_at = NULL, elapsed_seconds = 0 WHERE id = 1"
            )
        return self.stopwatch()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        total = int(seconds)
        hours, remainder = divmod(total, 3600)
        minutes, secs = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"


class ReminderScheduler:
    """Poll persistent reminders in the background while the desktop app runs."""

    def __init__(
        self,
        store: ProductivityStore,
        on_due: Callable[[list[dict[str, Any]]], None],
        interval_seconds: float = 1.0,
    ) -> None:
        self.store = store
        self.on_due = on_due
        self.interval_seconds = max(0.25, interval_seconds)
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="jarvis-reminder-scheduler",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def _run(self) -> None:
        while not self._stop_event.wait(self.interval_seconds):
            try:
                due_items = self.store.take_due_reminders()
            except (OSError, sqlite3.Error):
                logger.exception("Reminder scheduler could not check the local database")
                continue
            if due_items:
                try:
                    self.on_due(due_items)
                except Exception:
                    logger.exception("Reminder scheduler notification callback failed")
