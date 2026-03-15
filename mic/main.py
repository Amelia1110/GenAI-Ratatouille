"""UDP-triggered local speech capture backend for Project Ratatouille."""

from __future__ import annotations

import socket
import threading
from contextlib import closing
from dataclasses import dataclass

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

UDP_HOST = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 1024
PTT_START_MESSAGE = "PTT_START"
PTT_STOP_MESSAGE = "PTT_STOP"
START_MESSAGES = {PTT_START_MESSAGE, "START", "LISTEN", "PTT_DOWN", "PRESS"}
STOP_MESSAGES = {PTT_STOP_MESSAGE, "STOP", "PTT_UP", "RELEASE"}
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024
MODEL_NAME = "base.en"


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    format: int = FORMAT
    frames_per_buffer: int = FRAMES_PER_BUFFER


@dataclass
class AudioCaptureState:
    is_recording: bool = False
    frames: list[bytes] = None  # type: ignore[assignment]
    stream: pyaudio.Stream | None = None
    capture_thread: threading.Thread | None = None
    stop_capture: threading.Event = None  # type: ignore[assignment]
    lock: threading.Lock = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        self.frames = []
        self.stop_capture = threading.Event()
        self.lock = threading.Lock()


def create_udp_socket(host: str, port: int) -> socket.socket:
    """Create and bind a UDP socket for trigger packets."""
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_socket.bind((host, port))
    return udp_socket


def load_model_once() -> WhisperModel:
    """Load faster-whisper model once at startup."""
    try:
        model = WhisperModel(MODEL_NAME, device="cuda", compute_type="float16")
        print("Loaded faster-whisper on GPU.")
        return model
    except Exception:
        model = WhisperModel(MODEL_NAME, device="cpu", compute_type="int8")
        print("Loaded faster-whisper on CPU.")
        return model


def bytes_to_float32_mono(audio_bytes: bytes) -> np.ndarray:
    """Convert raw PCM16 bytes to float32 mono waveform in [-1, 1]."""
    audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
    if audio_i16.size == 0:
        return np.array([], dtype=np.float32)
    return (audio_i16.astype(np.float32) / 32768.0).flatten()


def capture_loop(state: AudioCaptureState, cfg: AudioConfig) -> None:
    """Continuously collect mic frames while recording is active."""
    assert state.stream is not None
    while not state.stop_capture.is_set():
        try:
            chunk = state.stream.read(cfg.frames_per_buffer, exception_on_overflow=False)
            state.frames.append(chunk)
        except OSError as exc:
            print(f"Microphone capture failed: {exc}")
            break


def start_recording(pa: pyaudio.PyAudio, cfg: AudioConfig, state: AudioCaptureState) -> None:
    """Begin recording if not already recording."""
    with state.lock:
        if state.is_recording:
            print("PTT start ignored: already recording.")
            return

        try:
            stream = pa.open(
                format=cfg.format,
                channels=cfg.channels,
                rate=cfg.sample_rate,
                input=True,
                frames_per_buffer=cfg.frames_per_buffer,
            )
        except OSError as exc:
            print(f"Could not open microphone stream: {exc}")
            return

        state.is_recording = True
        state.frames = []
        state.stop_capture.clear()
        state.stream = stream
        state.stream.start_stream()
        state.capture_thread = threading.Thread(
            target=capture_loop,
            args=(state, cfg),
            daemon=True,
        )
        state.capture_thread.start()
        print("PTT started: recording until PTT_STOP.")


def stop_recording(state: AudioCaptureState) -> bytes:
    """End recording if active and return captured bytes."""
    capture_thread: threading.Thread | None = None
    stream: pyaudio.Stream | None = None

    with state.lock:
        if not state.is_recording:
            return b""

        state.is_recording = False
        state.stop_capture.set()
        capture_thread = state.capture_thread
        stream = state.stream
        state.stream = None
        state.capture_thread = None

    if capture_thread is not None:
        capture_thread.join(timeout=2.0)

    if stream is not None:
        try:
            stream.stop_stream()
            stream.close()
        except OSError:
            pass

    with state.lock:
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


def run_server() -> None:
    """Start UDP trigger loop and run local faster-whisper on demand."""
    print("Initializing local model...")
    model = load_model_once()
    cfg = AudioConfig()
    capture_state = AudioCaptureState()
    pa = pyaudio.PyAudio()

    try:
        with closing(create_udp_socket(UDP_HOST, UDP_PORT)) as udp_socket:
            print(f"Listening for UDP push-to-talk packets on {UDP_HOST}:{UDP_PORT}")
            print(f"Start packet: {PTT_START_MESSAGE} (aliases: {sorted(START_MESSAGES)})")
            print(f"Stop packet: {PTT_STOP_MESSAGE} (aliases: {sorted(STOP_MESSAGES)})")
            while True:
                data, sender = udp_socket.recvfrom(BUFFER_SIZE)
                message = data.decode("utf-8", errors="ignore").strip()
                upper_message = message.upper()
                if message:
                    print(f"Received UDP from {sender}: {message}")

                if upper_message in START_MESSAGES:
                    start_recording(pa, cfg, capture_state)
                    continue

                if upper_message in STOP_MESSAGES:
                    audio_bytes = stop_recording(capture_state)
                    if not audio_bytes:
                        print("No audio captured (stop received while idle or silent start).")
                        continue

                    try:
                        text = transcribe_local(model, audio_bytes)
                        if text:
                            print(f"Transcribed: {text}")
                        else:
                            print("No intelligible speech detected.")
                    except Exception as exc:
                        print(f"Local transcription failed: {exc}")
                    continue

                if not message:
                    continue

                print(f"Ignored message from {sender}: {message}")
    except OSError as exc:
        print(f"UDP socket error: {exc}")
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        _ = stop_recording(capture_state)
        pa.terminate()


if __name__ == "__main__":
    run_server()
