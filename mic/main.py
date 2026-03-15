"""UDP-triggered local speech capture backend for Project Ratatouille."""

from __future__ import annotations

import socket
from contextlib import closing
from dataclasses import dataclass

import numpy as np
import pyaudio
from faster_whisper import WhisperModel

UDP_HOST = "0.0.0.0"
UDP_PORT = 5005
BUFFER_SIZE = 1024
TRIGGER_MESSAGE = "LISTEN"
SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024
MODEL_NAME = "base.en"
MAX_RECORD_SECONDS = 8


@dataclass
class AudioConfig:
    sample_rate: int = SAMPLE_RATE
    channels: int = CHANNELS
    format: int = FORMAT
    frames_per_buffer: int = FRAMES_PER_BUFFER
    max_record_seconds: int = MAX_RECORD_SECONDS


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


def capture_once(pa: pyaudio.PyAudio, cfg: AudioConfig) -> bytes:
    """Capture one fixed-length audio buffer from the default microphone."""
    total_chunks = int((cfg.sample_rate * cfg.max_record_seconds) / cfg.frames_per_buffer)

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
        return b""

    try:
        print("Trigger received. Listening...")
        frames: list[bytes] = []
        for _ in range(total_chunks):
            chunk = stream.read(cfg.frames_per_buffer, exception_on_overflow=False)
            frames.append(chunk)
        return b"".join(frames)
    except OSError as exc:
        print(f"Microphone capture failed: {exc}")
        return b""
    finally:
        try:
            stream.stop_stream()
            stream.close()
        except OSError:
            pass


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
    pa = pyaudio.PyAudio()

    try:
        with closing(create_udp_socket(UDP_HOST, UDP_PORT)) as udp_socket:
            print(f"Listening for UDP trigger '{TRIGGER_MESSAGE}' on {UDP_HOST}:{UDP_PORT}")
            while True:
                data, sender = udp_socket.recvfrom(BUFFER_SIZE)
                message = data.decode("utf-8", errors="ignore").strip()
                if message != TRIGGER_MESSAGE:
                    print(f"Ignored message from {sender}: {message}")
                    continue

                audio_bytes = capture_once(pa, cfg)
                if not audio_bytes:
                    print("No audio captured.")
                    continue

                try:
                    text = transcribe_local(model, audio_bytes)
                    if text:
                        print(f"Transcribed: {text}")
                    else:
                        print("No intelligible speech detected.")
                except Exception as exc:
                    print(f"Local transcription failed: {exc}")
    except OSError as exc:
        print(f"UDP socket error: {exc}")
    except KeyboardInterrupt:
        print("Server stopped.")
    finally:
        pa.terminate()


if __name__ == "__main__":
    run_server()
