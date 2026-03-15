"""Recent events and commentaries for context and dedupe."""
from typing import List


class RecentMemory:
    """Keep last N events and M commentaries for prompts and filtering."""

    def __init__(self, max_events: int = 10, max_commentaries: int = 5):
        self.max_events = max_events
        self.max_commentaries = max_commentaries
        self._events: List[str] = []
        self._commentaries: List[str] = []

    def add_events(self, events: List[str]) -> None:
        for e in events:
            self._events.append(e)
        if len(self._events) > self.max_events:
            self._events = self._events[-self.max_events :]

    def add_commentary(self, commentary: str) -> None:
        if commentary and commentary.strip():
            self._commentaries.append(commentary.strip())
        if len(self._commentaries) > self.max_commentaries:
            self._commentaries = self._commentaries[-self.max_commentaries :]

    def get_context(self) -> str:
        """Short context string for the prompt (recent commentaries)."""
        if not self._commentaries:
            return ""
        return " | ".join(self._commentaries[-3:])

    def get_recent_commentaries(self) -> List[str]:
        return list(self._commentaries)
