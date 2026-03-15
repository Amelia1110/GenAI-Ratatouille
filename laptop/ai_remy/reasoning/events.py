"""Extract list of cooking actions from Gemini ACTIONS block text."""
import re
from typing import List


def extract_events(actions_text: str) -> List[str]:
    """Parse ACTIONS: block into a list of action strings (one per line)."""
    if not actions_text or not actions_text.strip():
        return []
    lines = [ln.strip() for ln in actions_text.strip().splitlines() if ln.strip()]
    # Normalize: take first phrase if line has extra punctuation
    events = []
    for ln in lines:
        # Drop leading bullets/dashes
        ln = re.sub(r"^[\-\*]\s*", "", ln)
        if ln and ln.lower() != "none":
            events.append(ln)
    return events
