"""Kokoro TTS engine for local speaker playback.

Phase 4 goals:
- initialize Kokoro model once
- expose speak(text_chunk)
- optionally queue text chunks so LLM generation is not blocked by playback
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
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
        for key in ("audio", "samples", "wav", "waveform"):
            if key in candidate:
                return _coerce_audio_array(candidate[key])
        return np.array([], dtype=np.float32)

    if isinstance(candidate, (list, tuple)):
        if len(candidate) >= 3:
            audio_from_third = _coerce_audio_array(candidate[2])
            if audio_from_third.size:
                return audio_from_third

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
    """Placeholder hook for later network playback phases."""
    if not enabled or audio.size == 0:
        return

    STATIC_AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out_path = STATIC_AUDIO_DIR / "latest_tts.wav"
    sf.write(out_path, audio, sample_rate)


class KokoroTTSEngine:
    """Persistent Kokoro engine with blocking and queued speaking APIs."""

    def __init__(self, cfg: TTSConfig | None = None):
        self.cfg = cfg or TTSConfig()
        print("[tts] Initializing Kokoro pipeline (one-time load)...")
        self.pipeline = KPipeline(lang_code=self.cfg.lang_code)
        print("[tts] Kokoro pipeline ready.")

        self._queue: Queue[str] = Queue()
        self._stop_event = Event()
        self._worker_thread = Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _worker_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                text = self._queue.get(timeout=0.2)
            except Empty:
                continue

            try:
                self.speak(text)
            except Exception as exc:
                print(f"[tts] Playback failed: {exc}")
            finally:
                self._queue.task_done()

    def speak(self, text_chunk: str) -> None:
        """Blocking synthesis + playback for one chunk."""
        text = (text_chunk or "").strip()
        if not text:
            return

        chunks = iter_audio_chunks(self.pipeline, text, self.cfg.voice)
        audio = play_audio_chunks(chunks, self.cfg.sample_rate)
        save_audio_for_streaming_stub(audio, self.cfg.sample_rate, enabled=False)

    def enqueue(self, text_chunk: str) -> None:
        """Queue text chunk for background playback."""
        text = (text_chunk or "").strip()
        if text:
            self._queue.put(text)

    def close(self) -> None:
        """Stop worker thread."""
        self._stop_event.set()
        self._worker_thread.join(timeout=2.0)
