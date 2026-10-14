from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import MemorySaver

logger = logging.getLogger(__name__)


def create_memory_checkpointer() -> MemorySaver:
    """Create a LangGraph memory checkpointer for conversation state persistence.

    Uses in-memory checkpointing. For production, this can be swapped to
    SqliteSaver or PostgresSaver for durable persistence across restarts.

    Returns:
        A MemorySaver instance for the LangGraph compiled graph.
    """
    logger.info("Initializing in-memory conversation checkpointer")
    return MemorySaver()


class ConversationStore:
    """Lightweight local conversation history store (JSON file-based).

    Supplements LangGraph's checkpointer with a simple readable history
    that survives process restarts. Each conversation is stored as a
    JSON file in the conversations directory.
    """

    def __init__(self, storage_dir: str | Path | None = None) -> None:
        if storage_dir is None:
            import os
            local_data = Path(os.getenv("LOCALAPPDATA", str(Path.home() / "AppData/Local")))
            storage_dir = local_data / "JARVIS" / "conversations"
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Conversation store: %s", self.storage_dir)

    def save_turn(self, thread_id: str, role: str, content: str) -> None:
        """Append a message to a conversation's history file."""
        filepath = self.storage_dir / f"{thread_id}.json"
        history = self._load_file(filepath)
        history.append({"role": role, "content": content})
        # Keep last 100 messages per conversation
        if len(history) > 100:
            history = history[-100:]
        filepath.write_text(json.dumps(history, indent=2, ensure_ascii=False), encoding="utf-8")

    def get_history(self, thread_id: str, limit: int = 20) -> list[dict[str, str]]:
        """Retrieve recent messages for a conversation thread."""
        filepath = self.storage_dir / f"{thread_id}.json"
        history = self._load_file(filepath)
        return history[-limit:]

    def list_threads(self) -> list[str]:
        """List all conversation thread IDs."""
        return [f.stem for f in self.storage_dir.glob("*.json")]

    def clear_thread(self, thread_id: str) -> bool:
        """Delete a conversation thread."""
        filepath = self.storage_dir / f"{thread_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False

    @staticmethod
    def _load_file(filepath: Path) -> list[dict[str, str]]:
        if filepath.exists():
            try:
                return json.loads(filepath.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                return []
        return []
