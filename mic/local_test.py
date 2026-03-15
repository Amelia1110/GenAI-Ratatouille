"""True push-to-talk local transcription using faster-whisper."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from pynput import keyboard


SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024
MODEL_NAME = "base.en"  # or "small.en"


@dataclass
class AudioCaptureState:
    is_recording: bool = False
    frames: list[bytes] = field(default_factory=list)
    stream: pyaudio.Stream | None = None
    capture_thread: threading.Thread | None = None
    stop_capture: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


def load_model_once() -> WhisperModel:
    """Load faster-whisper model once at startup."""
    try:
        # Try GPU first if available.
        model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
        print("Loaded faster-whisper on GPU.")
        return model
    except Exception:
        # Safe fallback for CPU-only laptops.
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        print("Loaded faster-whisper on CPU.")
        return model


def bytes_to_float32_mono(audio_bytes: bytes) -> np.ndarray:
    """Convert raw PCM16 bytes to float32 mono waveform in [-1, 1]."""
    audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio_i16.size == 0:
        return np.array([], dtype=np.float32)
    return (audio_i16.astype(np.float32) / 32768.0).flatten()


def capture_loop(state: AudioCaptureState) -> None:
    """Continuously read audio chunks while recording is active."""
    assert state.stream is not None
    while not state.stop_capture.is_set():
        try:
            chunk = state.stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
            state.frames.append(chunk)
        except OSError as exc:
            print(f"Audio stream read error: {exc}")
            break


def start_recording(pa: pyaudio.PyAudio, state: AudioCaptureState) -> None:
    """Start audio stream only once per Enter key hold."""
    with state.lock:
        if state.is_recording:
            return  # Debounce key-repeat while Enter is held.

        try:
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                frames_per_buffer=FRAMES_PER_BUFFER,
            )
        except OSError as exc:
            print(f"Could not open microphone stream: {exc}")
            return

        state.is_recording = True
        state.frames = []
        state.stop_capture.clear()
        state.stream = stream
        state.capture_thread = threading.Thread(target=capture_loop, args=(state,), daemon=True)
        state.capture_thread.start()
        print("Recording... release Enter to transcribe.")


def stop_recording(state: AudioCaptureState) -> bytes:
    """Stop stream and return collected audio bytes."""
    with state.lock:
        if not state.is_recording:
            return b""

        state.is_recording = False
        state.stop_capture.set()

        if state.capture_thread is not None:
            state.capture_thread.join(timeout=2.0)

        if state.stream is not None:
            try:
                state.stream.stop_stream()
                state.stream.close()
            except OSError:
                pass

        state.stream = None
        state.capture_thread = None
        audio_bytes = b"".join(state.frames)
        state.frames = []
        return audio_bytes


def transcribe_local(model: WhisperModel, audio_bytes: bytes) -> str:
    """Run local faster-whisper transcription."""
    waveform = bytes_to_float32_mono(audio_bytes)
    if waveform.size == 0:
        return ""

    segments, _info = model.transcribe(
        waveform,
        language="en",
        vad_filter=True,
        beam_size=5,
        temperature=0.0,
    )
    text_parts = [segment.text.strip() for segment in segments if segment.text.strip()]
    return " ".join(text_parts).strip()


def is_quit_key(key: object) -> bool:
    """Return True when user pressed a key that should end the program."""
    return key == keyboard.Key.esc


def run_push_to_talk() -> None:
    """Main loop: hold Enter to record, release to transcribe, repeat."""
    print("Initializing local model...")
    model = load_model_once()

    pa = pyaudio.PyAudio()
    state = AudioCaptureState()
    exiting = threading.Event()
    enter_is_down = threading.Event()
    listener: keyboard.Listener | None = None

    print("Waiting for Enter key hold. Press Esc to exit.")

    def on_press(key: object) -> None:
        if is_quit_key(key):
            if exiting.is_set():
                return
            exiting.set()
            print("Exit requested. Stopping...")
            _ = stop_recording(state)
            if listener is not None:
                listener.stop()
            return

        if key == keyboard.Key.enter:
            if enter_is_down.is_set():
                return
            enter_is_down.set()
            start_recording(pa, state)

    def on_release(key: object) -> None:
        if key != keyboard.Key.enter:
            return

        if not enter_is_down.is_set():
            return
        enter_is_down.clear()

        audio_bytes = stop_recording(state)
        if not audio_bytes:
            print("No audio captured.")
            print("Waiting for Enter key hold...")
            return

        try:
            text = transcribe_local(model, audio_bytes)
            if text:
                print(f"Transcribed: {text}")
            else:
                print("No intelligible speech detected.")
        except Exception as exc:
            print(f"Local transcription failed: {exc}")

        print("Waiting for Enter key hold...")

    try:
        listener = keyboard.Listener(
            on_press=on_press,
            on_release=on_release,
            suppress=True,
        )
        listener.start()
        while listener.is_alive() and not exiting.is_set():
            time.sleep(0.05)
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        # Ensure cleanup if script exits during recording.
        _ = stop_recording(state)
        if listener is not None and listener.is_alive():
            listener.stop()
        pa.terminate()


if __name__ == "__main__":
    run_push_to_talk()