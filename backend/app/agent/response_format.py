from __future__ import annotations

import re
from collections.abc import Callable
from difflib import SequenceMatcher


class AssistantResponseStream:
    """Parse JARVIS's internal response envelope without exposing its markup."""

    _opening = re.compile(r"<\s*([a-z][\w-]*)\b([^>]*)>", re.IGNORECASE)
    # Permit whitespace and stray attributes on envelope tags. Small local
    # models often emit `<summary >` or `<displaydetails class="...">` even
    # though the prompt asks for plain tags.
    _tag = re.compile(r"<\s*(/?)\s*([a-z][\w-]*)\b([^>]*)>", re.IGNORECASE)
    _partial_tag = re.compile(r"<\s*/?\s*([a-z][\w-]*)\b[^>]*$", re.IGNORECASE)
    _attribute = re.compile(
        r"([a-z][\w-]*)\s*=\s*(['\"]?)(true|false|yes|no|1|0)\2",
        re.IGNORECASE,
    )
    _closing_tail = re.compile(r"</?\s*[a-z][\w-]*\s*$", re.IGNORECASE)

    def __init__(
        self,
        on_summary: Callable[[str], None] | None = None,
        on_details_start: Callable[[], None] | None = None,
        on_details: Callable[[str], None] | None = None,
        on_summary_complete: Callable[[], None] | None = None,
    ) -> None:
        self.on_summary = on_summary or (lambda _text: None)
        self.on_details_start = on_details_start or (lambda: None)
        self.on_details = on_details or (lambda _text: None)
        self.on_summary_complete = on_summary_complete or (lambda: None)
        self.buffer = ""
        self.state = "opening"
        self.passthrough = False
        self.valid = False
        self.is_detail_insight_needed = False
        self.summary_complete = False
        self.details_started = False
        self.details_complete = False
        self.summary_parts: list[str] = []
        self.details_parts: list[str] = []

    def feed(self, chunk: str) -> None:
        if not chunk or self.state == "done":
            return
        if self.passthrough:
            self.buffer += chunk
            self._drain_plain_markup()
            return
        self.buffer += chunk
        self._drain()

    def finish(self) -> None:
        """Flush partial or unstructured output, stripping schema-like tags."""
        if self.state == "done":
            return
        if self.passthrough:
            self._drain_plain_markup(final=True)
            self.state = "done"
            return
        self._drain()
        if self.valid:
            remainder = self._clean_markup(self.buffer)
            self.buffer = ""
            if self.state == "summary" and not self.summary_complete:
                self._emit_summary(remainder)
                self.summary_complete = True
                self.on_summary_complete()
            elif self.state == "details" and self.is_detail_insight_needed:
                self._emit_details(remainder)
            self.state = "done"
            return
        if self.buffer:
            self._emit_summary(self._clean_markup(self.buffer))
            self.buffer = ""
        self.state = "done"

    def _drain(self) -> None:
        while True:
            if self.state == "opening":
                match = next(
                    (
                        candidate
                        for candidate in self._opening.finditer(self.buffer)
                        if self._similar(candidate.group(1), "response") >= 0.78
                    ),
                    None,
                )
                if match is None:
                    candidate = self.buffer.lstrip()
                    if candidate and not candidate.startswith("<"):
                        # Plain legacy responses can still stream immediately.
                        pending = self.buffer
                        self.buffer = ""
                        self.passthrough = True
                        self.buffer = pending
                        self._drain_plain_markup()
                    return
                self.is_detail_insight_needed = self._read_details_flag(match.group(2))
                self.valid = True
                self.buffer = self.buffer[match.end():]
                self.state = "summary_open"
            elif self.state == "summary_open":
                found = self._find_tag(self.buffer, "summary", closing=False)
                if found is None:
                    self.buffer = self.buffer[-(len("<summary>") - 1):]
                    return
                _, end = found
                self.buffer = self.buffer[end:]
                self.state = "summary"
            elif self.state == "summary":
                found = self._find_tag(self.buffer, "summary", closing=True)
                if found is None:
                    safe_length = max(0, len(self.buffer) - len("</summary>") + 1)
                    self._emit_summary(self.buffer[:safe_length])
                    self.buffer = self.buffer[safe_length:]
                    return
                position, end = found
                self._emit_summary(self.buffer[:position])
                self.buffer = self.buffer[end:]
                self.summary_complete = True
                self.on_summary_complete()
                self.state = "details_open"
            elif self.state == "details_open":
                found = self._find_tag(self.buffer, "display_details", closing=False)
                if found is None:
                    # Most often the model omitted the optional empty details block.
                    self.buffer = self.buffer[-(len("<display_details>") - 1):]
                    return
                _, end = found
                self.buffer = self.buffer[end:]
                self.state = "details"
                if self.is_detail_insight_needed:
                    self.details_started = True
                    self.on_details_start()
            elif self.state == "details":
                found = self._find_tag(self.buffer, "display_details", closing=True)
                if found is None:
                    safe_length = max(0, len(self.buffer) - len("</display_details>") + 1)
                    self._emit_details(self.buffer[:safe_length])
                    self.buffer = self.buffer[safe_length:]
                    return
                position, end = found
                self._emit_details(self.buffer[:position])
                self.buffer = self.buffer[end:]
                self.details_complete = True
                self.state = "done"
                return
            else:
                return

    def _emit_summary(self, text: str) -> None:
        if text:
            self.summary_parts.append(text)
            self.on_summary(text)

    def _emit_details(self, text: str) -> None:
        if text and self.is_detail_insight_needed:
            self.details_parts.append(text)
            self.on_details(text)

    def _drain_plain_markup(self, *, final: bool = False) -> None:
        """Stream plain output while filtering schema tags, even across chunks.

        This also covers a model that begins with a preamble, which otherwise
        could make the parser enter plain-text mode before the response envelope
        arrives and leak the later tags directly into the desktop chat.
        """
        output: list[str] = []
        cursor = 0
        for match in self._tag.finditer(self.buffer):
            output.append(self.buffer[cursor:match.start()])
            if not self._is_schema_tag(match.group(2)):
                output.append(match.group(0))
            cursor = match.end()

        if final:
            output.append(self._clean_markup(self.buffer[cursor:]))
            self.buffer = ""
        else:
            partial_start = self.buffer.rfind("<")
            has_partial_tag = (
                partial_start >= cursor
                and ">" not in self.buffer[partial_start:]
            )
            if has_partial_tag:
                output.append(self.buffer[cursor:partial_start])
                self.buffer = self.buffer[partial_start:]
            else:
                output.append(self.buffer[cursor:])
                self.buffer = ""
        self._emit_summary("".join(output))

    @classmethod
    def _similar(cls, candidate: str, expected: str) -> float:
        compact_candidate = re.sub(r"[^a-z0-9]", "", candidate.lower())
        compact_expected = re.sub(r"[^a-z0-9]", "", expected.lower())
        return SequenceMatcher(None, compact_candidate, compact_expected).ratio()

    @classmethod
    def _read_details_flag(cls, attributes: str) -> bool:
        for match in cls._attribute.finditer(attributes):
            if cls._similar(match.group(1), "is_detail_insight_needed") >= 0.70:
                return match.group(3).lower() in {"true", "yes", "1"}
        # Missing or unrecognized flags fail closed: details are not displayed.
        return False

    @classmethod
    def _find_tag(
        cls, text: str, expected: str, *, closing: bool
    ) -> tuple[int, int] | None:
        for match in cls._tag.finditer(text):
            if bool(match.group(1)) != closing:
                continue
            if cls._similar(match.group(2), expected) >= 0.72:
                return match.start(), match.end()
        return None

    @classmethod
    def _clean_markup(cls, text: str) -> str:
        def replace_opening(match: re.Match[str]) -> str:
            return "" if cls._is_schema_tag(match.group(1)) else match.group(0)

        def replace_tag(match: re.Match[str]) -> str:
            return "" if cls._is_schema_tag(match.group(2)) else match.group(0)

        cleaned = cls._opening.sub(replace_opening, text)
        cleaned = cls._tag.sub(replace_tag, cleaned)
        cleaned = cls._partial_tag.sub(
            lambda match: "" if cls._is_schema_tag(match.group(1)) else match.group(0),
            cleaned,
        )
        cleaned = cls._closing_tail.sub("", cleaned)
        return re.sub(r"\n{3,}", "\n\n", cleaned).strip()

    @classmethod
    def _is_schema_tag(cls, tag_name: str) -> bool:
        return any(
            cls._similar(tag_name, expected) >= 0.72
            for expected in ("response", "summary", "display_details")
        )

    @property
    def summary(self) -> str:
        return "".join(self.summary_parts).strip()

    @property
    def details(self) -> str:
        return "".join(self.details_parts).strip()
