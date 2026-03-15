"""
Vocal-only recipe flow: get the dish from the user via microphone (no typing).
Remy asks out loud; user responds by voice.
"""

import sys
from pathlib import Path

_laptop = Path(__file__).resolve().parent.parent
if str(_laptop) not in sys.path:
    sys.path.insert(0, str(_laptop))

from ai_remy import config

# Phrase Remy speaks at session start (must be spoken via TTS before calling get_dish_from_mic).
PROMPT_DISH = "What dish are we making today?"


def get_dish_from_mic(listen_seconds: int = 5) -> str:
    """
    Listen to the microphone and return what the user said (dish name, e.g. "Pasta").
    Vocal-only: no text fallback. Returns empty string if STT unavailable or fails.
    """
    try:
        import speech_recognition as sr
    except ImportError:
        return ""

    r = sr.Recognizer()
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source, duration=0.5)
        try:
            audio = r.record(source, duration=listen_seconds)
        except Exception:
            return ""

    try:
        text = r.recognize_google(audio)
        return (text or "").strip()
    except Exception:
        return ""


def get_recipe(use_mic: bool = False) -> str:
    """
    Get dish name: from RECIPE env only (for headless/test), or from mic.
    For a fully vocal session, run with use_mic=True and ask the question via TTS first.
    """
    if config.RECIPE:
        return config.RECIPE.strip()
    if use_mic:
        return get_dish_from_mic()
    return ""
