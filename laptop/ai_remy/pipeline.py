"""
Single entry point for the Gemini pipeline (no OpenCV).
Call from the other branch with a frame (numpy array or image bytes).
"""
from typing import List, Tuple, Union

import numpy as np

from ai_remy.state.memory import RecentMemory
from ai_remy.vision.preprocess import preprocess_frame
from ai_remy.vision.gemini_client import analyze_scene
from ai_remy.reasoning.events import extract_events
from ai_remy.reasoning.filter import should_speak


def process_frame(
    image_input: Union[bytes, np.ndarray],
    memory: RecentMemory,
) -> Tuple[List[str], str, bool]:
    """
    Run preprocess -> Gemini (scene + actions + comment) -> events -> filter.

    Returns:
        events: list of detected cooking actions
        commentary: one short Remy-style line
        should_speak: True if the caller should speak the commentary (not empty, not duplicate)
    """
    image_bytes = preprocess_frame(image_input)
    scene_text, actions_text, commentary = analyze_scene(
        image_bytes,
        recent_context=memory.get_context(),
    )
    events = extract_events(actions_text)
    memory.add_events(events)

    if not should_speak(commentary, memory.get_recent_commentaries()):
        return (events, commentary or "", False)

    memory.add_commentary(commentary)
    return (events, commentary, True)
