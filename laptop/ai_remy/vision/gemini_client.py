"""
Gemini: single call for scene description, actions, and one Remy-style comment.
Uses GEMINI_MODEL from config (default gemini-2.5-flash). Run list_models.py to see available names.
"""
import re
from typing import List, Tuple

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

    text = response.text.strip()
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
        # COMMENT might be missing; take rest after ACTIONS
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
