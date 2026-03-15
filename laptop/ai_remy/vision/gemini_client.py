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

# Base instruction for mentor mode: guide the user through their recipe and comment on their actions.
SCENE_PROMPT = """You are Remy, a warm and knowledgeable cooking mentor. You are watching the cook via a live kitchen camera. Your job is to guide them through the recipe they are making, comment on what they are doing, and gently steer them (e.g. suggest the next step, warn if something is off, or praise good technique).

First describe in 1-2 sentences what you see in the scene (ingredients, tools, actions).
Then list cooking-related actions you see, one per line, from this set (use only these when they apply):
- chopping onion / chopping vegetables / cutting
- stirring pan / stirring
- heating oil / oil in pan
- adding ingredients to pan
- seasoning / adding salt or spices
- flipping or turning food
- no significant cooking action

Then give exactly ONE short spoken line (max 20 words) as Remy. The line should:
- Comment on what they are doing (e.g. "Nice, the onions are in. Next, get the pan hot for the garlic.")
- If you know their recipe: guide them toward the next step or praise when they are on track.
- Be encouraging, observant, and occasionally playful. Never condescending.
Do not use markdown or quotes.

Format your response exactly as:
SCENE: <your 1-2 sentence description>
ACTIONS:
<action 1>
<action 2>
COMMENT: <one short line to be spoken>
"""


def _image_part(jpeg_bytes: bytes):
    return {"inline_data": {"mime_type": "image/jpeg", "data": jpeg_bytes}}


RECIPE_GENERATION_PROMPT = """You are a friendly cooking mentor. The user said they want to make: "{dish}".

Generate a concise step-by-step recipe for that dish. Use clear, short steps (one or two sentences each). Number the steps. Do not include a long intro—just the steps the cook will follow. Output only the recipe steps, no other text."""


DISH_SUMMARY_PROMPT = """You are helping normalize a cooking session title.
User originally said: "{user_input}"

Recipe steps generated:
{recipe_steps}

Return exactly one short line (max 12 words) summarizing what dish this is.
Do not include quotes, markdown, numbering, or extra explanation."""


def generate_recipe_for_dish(dish: str) -> str:
    """Generate a step-by-step recipe for the given dish (text-only Gemini call). Returns recipe text or empty on failure."""
    if not (dish or "").strip():
        return ""
    prompt = RECIPE_GENERATION_PROMPT.format(dish=dish.strip())
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    response = model.generate_content(prompt)
    if not response or not response.text:
        return ""
    return response.text.strip()


def summarize_dish_from_recipe(user_input: str, recipe_steps: str) -> str:
    """Return a concise dish summary inferred from the generated recipe."""
    cleaned_input = (user_input or "").strip()
    cleaned_steps = (recipe_steps or "").strip()
    if not cleaned_input and not cleaned_steps:
        return ""

    prompt = DISH_SUMMARY_PROMPT.format(
        user_input=cleaned_input or "(none)",
        recipe_steps=cleaned_steps or "(none)",
    )
    model = genai.GenerativeModel(config.GEMINI_MODEL)
    response = model.generate_content(prompt)
    if response and response.text:
        return response.text.strip()
    return cleaned_input


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
    recipe: str = "",
    recipe_steps: str = "",
    recent_events_summary: str = "",
    user_prompt: str = "",
    conversation_context: str = "",
) -> Tuple[str, str, str]:
    """
    Single Gemini call: scene + actions + one mentor comment.
    recipe: dish name. recipe_steps: full AI-generated recipe for step-by-step guidance.
    Returns: (scene_text, actions_text, commentary)
    """
    prompt = SCENE_PROMPT
    prefix_parts = []
    if recipe:
        prefix_parts.append("The cook is making: " + recipe + ".")
    if recipe_steps:
        prefix_parts.append("Recipe to guide them through:\n" + recipe_steps)
    if recent_events_summary:
        prefix_parts.append("Recent actions you've seen: " + recent_events_summary + ".")
    if recent_context:
        prefix_parts.append("What you (Remy) already said recently: " + recent_context + ".")
    if conversation_context:
        prefix_parts.append("Recent user/remy conversation turns:\n" + conversation_context)
    if user_prompt:
        prefix_parts.append("The user just said (voice): " + user_prompt)
    if prefix_parts:
        prompt = "\n".join(prefix_parts) + "\n\n" + prompt

    model = genai.GenerativeModel(config.GEMINI_MODEL)
    part = _image_part(image_bytes)
    response = model.generate_content([prompt, part])
    if not response or not response.text:
        return ("", "", "")

    return _parse_response_text(response.text.strip())


def analyze_scene_stream(
    image_bytes: bytes,
    recent_context: str = "",
    recipe: str = "",
    recipe_steps: str = "",
    recent_events_summary: str = "",
    on_comment_chunk: Callable[[str], None] | None = None,
    user_prompt: str = "",
    conversation_context: str = "",
) -> Tuple[str, str, str]:
    """Streaming Gemini call that emits commentary sentence chunks as they complete."""
    prompt = SCENE_PROMPT
    prefix_parts = []
    if recipe:
        prefix_parts.append("The cook is making: " + recipe + ".")
    if recipe_steps:
        prefix_parts.append("Recipe to guide them through:\n" + recipe_steps)
    if recent_events_summary:
        prefix_parts.append("Recent actions you've seen: " + recent_events_summary + ".")
    if recent_context:
        prefix_parts.append("What you (Remy) already said recently: " + recent_context + ".")
    if conversation_context:
        prefix_parts.append("Recent user/remy conversation turns:\n" + conversation_context)
    if user_prompt:
        prefix_parts.append("The user just said (voice): " + user_prompt)
    if prefix_parts:
        prompt = "\n".join(prefix_parts) + "\n\n" + prompt

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
