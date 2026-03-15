"""Local terminal-driven Kokoro TTS smoke test for Project Ratatouille.

This script intentionally keeps everything local:
- input text from terminal
- synthesize speech with Kokoro
- play audio through laptop speakers

Future phases will replace input and output plumbing while keeping synthesis code reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import sounddevice as sd
import soundfile as sf
from kokoro import KPipeline

DEFAULT_LANG_CODE = "a"  # American English
DEFAULT_VOICE = "am_adam"
DEFAULT_SAMPLE_RATE = 24000
STATIC_AUDIO_DIR = Path("static/audio")


@dataclass(frozen=True)
class TTSConfig:
    """Configuration for local Kokoro synthesis and playback."""

    lang_code: str = DEFAULT_LANG_CODE
    voice: str = DEFAULT_VOICE
    sample_rate: int = DEFAULT_SAMPLE_RATE


def initialize_pipeline(cfg: TTSConfig) -> KPipeline:
    """Load Kokoro once at startup to avoid per-request model latency."""
    print("Initializing Kokoro pipeline (one-time load)...")
    pipeline = KPipeline(lang_code=cfg.lang_code)
    print("Kokoro pipeline ready.")
    return pipeline


def get_next_text() -> str:
    """Read text from terminal for local/manual testing.

    NOTE: Future integration point:
    Replace this terminal input with the Gemini LLM text stream emitter.
    """
    return input("Enter text (or 'quit' to exit): ").strip()


def _coerce_audio_array(candidate: Any) -> np.ndarray:
    """Return a 1D float32 audio array from a Kokoro output candidate."""
    if candidate is None:
        return np.array([], dtype=np.float32)

    if hasattr(candidate, "audio"):
        return _coerce_audio_array(getattr(candidate, "audio"))

    if isinstance(candidate, np.ndarray):
        return candidate.astype(np.float32).flatten()

    if hasattr(candidate, "detach") and hasattr(candidate, "cpu") and hasattr(candidate, "numpy"):
        return candidate.detach().cpu().numpy().astype(np.float32).flatten()

    if isinstance(candidate, dict):
        # Common keys used by audio payload dictionaries.
        for key in ("audio", "samples", "wav", "waveform"):
            if key in candidate:
                return _coerce_audio_array(candidate[key])
        return np.array([], dtype=np.float32)

    if isinstance(candidate, (list, tuple)):
        # Kokoro pipelines may yield [graphemes, phonemes, audio] or similar.
        if len(candidate) >= 3:
            audio_from_third = _coerce_audio_array(candidate[2])
            if audio_from_third.size:
                return audio_from_third

        # Fallback: find the first element that can be interpreted as a numeric audio array.
        for element in candidate:
            audio_from_element = _coerce_audio_array(element)
            if audio_from_element.size:
                return audio_from_element
        return np.array([], dtype=np.float32)

    try:
        return np.asarray(candidate, dtype=np.float32).flatten()
    except (TypeError, ValueError):
        return np.array([], dtype=np.float32)


def iter_audio_chunks(pipeline: KPipeline, text: str, voice: str) -> Iterable[np.ndarray]:
    """Yield audio chunks from Kokoro for the given text."""
    generator = pipeline(text, voice=voice)
    for item in generator:
        audio_np = _coerce_audio_array(item)
        if audio_np.size:
            yield audio_np


def play_audio_chunks(chunks: Iterable[np.ndarray], sample_rate: int) -> np.ndarray:
    """Concatenate and play chunks through local speakers."""
    all_chunks = list(chunks)
    if not all_chunks:
        return np.array([], dtype=np.float32)

    audio = np.concatenate(all_chunks)
    sd.play(audio, samplerate=sample_rate)
    sd.wait()
    return audio


def save_audio_for_streaming_stub(audio: np.ndarray, sample_rate: int, enabled: bool = False) -> None:
    """Placeholder hook for upcoming ESP32 streaming architecture.

    NOTE: Future integration point:
    Replace local playback flow with:
    1) write synthesized audio to a static HTTP-served folder
    2) trigger ESP32 UDP playback stream

    `enabled` remains False for this phase so no network/server behavior is introduced.
    """
    if not enabled or audio.size == 0:
        return

    STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATIC_AUDIO_DIR / "latest_tts.wav"
    sf.write(out_path, audio, sample_rate)


def run_local_tts_loop(cfg: TTSConfig) -> None:
    """Interactive local loop for manual TTS testing."""
    pipeline = initialize_pipeline(cfg)

    while True:
        text = get_next_text()
        if not text:
            print("Empty input, please type some text.")
            continue
        if text.lower() in {"quit", "exit"}:
            print("Exiting local Kokoro test.")
            break

        try:
            chunks = iter_audio_chunks(pipeline, text, cfg.voice)
            audio = play_audio_chunks(chunks, cfg.sample_rate)
            save_audio_for_streaming_stub(audio, cfg.sample_rate, enabled=False)
        except Exception as exc:
            print(f"TTS failed: {exc}")


if __name__ == "__main__":
    run_local_tts_loop(TTSConfig())
