"""Commentary filter: avoid repeating the same or empty comments."""
from typing import List


def should_speak(commentary: str, recent_commentaries: List[str]) -> bool:
    """Return True if we should speak this comment (non-empty, not too similar to recent)."""
    if not commentary or not commentary.strip():
        return False
    c = commentary.strip().lower()
    if len(c) < 3:
        return False
    # Simple dedupe: if any recent comment is very similar, skip
    for prev in recent_commentaries[-5:]:
        if not prev:
            continue
        p = prev.strip().lower()
        if c == p or (len(c) > 10 and p in c) or (len(p) > 10 and c in p):
            return False
    return True
