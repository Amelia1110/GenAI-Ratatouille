"""
Gemini: single call for scene description, actions, and one Remy-style comment.
Uses GEMINI_MODEL from config (default gemini-2.5-flash). Run list_models.py to see available names.
"""
import re
from typing import Callable, Tuple

import google.generativeai as genai

from ai_remy import config

# Configure once
if config.GEMINI_API_KEY:
    genai.configure(api_key=config.GEMINI_API_KEY)

SCENE_PROMPT = """You are analyzing a live kitchen camera frame from a cooking session.

Describe in 1-2 sentences what is happening in the scene (ingredients, tools, actions).
Then list cooking-related actions you see, one per line, from this set (use only these when they apply):
- chopping onion / chopping vegetables / cutting
- stirring pan / stirring
- heating oil / oil in pan
- adding ingredients to pan
- seasoning / adding salt or spices
- flipping or turning food
- no significant cooking action

Then give exactly ONE short spoken comment (max 15 words) as Remy, a friendly cooking mentor: encouraging, observant, occasionally playful. Do not use markdown or quotes.

Format your response exactly as:
SCENE: <your 1-2 sentence description>
ACTIONS:
<action 1>
<action 2>
COMMENT: <one short line to be spoken>
"""


def _image_part(jpeg_bytes: bytes):
    return {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_bytes}}


def _split_complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Split by sentence boundary while preserving trailing incomplete fragment."""
    parts = re.split(r"(?<=[.!?])\s+", buffer)
    if len(parts) <= 1:
        return [], buffer
    completed = [segment.strip() for segment in parts[:-1] if segment.strip()]
    remainder = parts[-1]
    return completed, remainder


def _parse_response_text(text: str) -> Tuple[str, str, str]:
    scene = ""
    actions_block = ""
    comment = ""

    scene_m = re.search(r"SCENE:\s*(.+?)(?=ACTIONS:)", text, re.DOTALL | re.IGNORECASE)
    if scene_m:
        scene = scene_m.group(1).strip()

    actions_m = re.search(r"ACTIONS:\s*(.+?)(?=COMMENT:)", text, re.DOTALL | re.IGNORECASE)
    if actions_m:
        actions_block = actions_m.group(1).strip()
    else:
        actions_m = re.search(r"ACTIONS:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
        if actions_m:
            block = actions_m.group(1).strip()
            if "COMMENT:" in block:
                parts = re.split(r"COMMENT:\s*", block, maxsplit=1, flags=re.IGNORECASE)
                actions_block = parts[0].strip()
                comment = parts[1].strip() if len(parts) > 1 else ""
            else:
                actions_block = block

    comment_m = re.search(r"COMMENT:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if comment_m:
        comment = comment_m.group(1).strip().split("\n")[0]

    return (scene, actions_block, comment)


def analyze_scene(
    image_bytes: bytes,
    recent_context: str = "",
) -> Tuple[str, str, str]:
    """
    Single Gemini call: scene + actions + one comment.
    Returns: (scene_text, actions_text, commentary)
    """
    prompt = SCENE_PROMPT
    if recent_context:
        prompt = "Recent context (what you already said): " + recent_context + "\n\n" + prompt

    model = genai.GenerativeModel(config.GEMINI_MODEL)
    part = _image_part(image_bytes)
    response = model.generate_content([prompt, part])
    if not response or not response.text:
        return ("", "", "")

    return _parse_response_text(response.text.strip())


def analyze_scene_stream(
    image_bytes: bytes,
    recent_context: str = "",
    on_comment_chunk: Callable[[str], None] | None = None,
) -> Tuple[str, str, str]:
    """Streaming Gemini call that emits commentary sentence chunks as they complete."""
    prompt = SCENE_PROMPT
    if recent_context:
        prompt = "Recent context (what you already said): " + recent_context + "\n\n" + prompt

    model = genai.GenerativeModel(config.GEMINI_MODEL)
    part = _image_part(image_bytes)
    response_stream = model.generate_content([prompt, part], stream=True)

    full_text = ""
    comment_buffer = ""
    delivered_until = 0

    for chunk in response_stream:
        delta = getattr(chunk, "text", "")
        if not delta:
            continue
        full_text += delta

        marker = re.search(r"COMMENT:\s*", full_text, re.IGNORECASE)
        if not marker:
            continue

        comment_so_far = full_text[marker.end() :]
        if len(comment_so_far) <= delivered_until:
            continue

        comment_buffer += comment_so_far[delivered_until:]
        delivered_until = len(comment_so_far)

        completed, comment_buffer = _split_complete_sentences(comment_buffer)
        if on_comment_chunk:
            for sentence in completed:
                on_comment_chunk(sentence)

    tail = comment_buffer.strip()
    if tail and on_comment_chunk:
        on_comment_chunk(tail)

    return _parse_response_text(full_text.strip())
