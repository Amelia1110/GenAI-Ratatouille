"""AI Remy multimodal runner: camera frames + local push-to-talk STT + Gemini."""

from __future__ import annotations

import argparse
from collections.abc import Callable
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pyaudio
from faster_whisper import WhisperModel
from pynput import keyboard

# Ensure laptop/ is on path for ai_remy
_laptop_dir = Path(__file__).resolve().parent
if str(_laptop_dir) not in sys.path:
    sys.path.insert(0, str(_laptop_dir))

from ai_remy.pipeline import process_frame_streaming
from ai_remy.state.memory import RecentMemory
from ai_remy.tts_engine import KokoroTTSEngine
from ai_remy.vision.gemini_client import generate_recipe_for_dish, summarize_dish_from_recipe

SAMPLE_RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
FRAMES_PER_BUFFER = 1024


@dataclass
class AudioCaptureState:
    is_recording: bool = False
    frames: list[bytes] = field(default_factory=list)
    stream: pyaudio.Stream | None = None
    capture_thread: threading.Thread | None = None
    stop_capture: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)


class LatestFrameBuffer:
    def __init__(self, frame_path: Path, poll_interval: float = 0.2):
        self.frame_path = frame_path
        self.poll_interval = poll_interval
        self._lock = threading.Lock()
        self._latest_bytes: bytes = b""
        self._latest_mtime: float | None = None
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=2.0)

    def get_latest_frame(self) -> tuple[bytes, float | None]:
        with self._lock:
            return self._latest_bytes, self._latest_mtime

    def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if not self.frame_path.exists():
                    time.sleep(self.poll_interval)
                    continue
                mtime = self.frame_path.stat().st_mtime
                with self._lock:
                    known_mtime = self._latest_mtime
                if known_mtime is not None and mtime <= known_mtime:
                    time.sleep(self.poll_interval)
                    continue
                frame_bytes = self.frame_path.read_bytes()
                if not frame_bytes:
                    time.sleep(self.poll_interval)
                    continue
                with self._lock:
                    self._latest_bytes = frame_bytes
                    self._latest_mtime = mtime
            except OSError:
                time.sleep(self.poll_interval)


class PushToTalkSTT:
    def __init__(
        self,
        model_name: str,
        before_listen: Callable[[], None] | None = None,
        after_listen: Callable[[], None] | None = None,
    ):
        self.model_name = model_name
        self.before_listen = before_listen
        self.after_listen = after_listen
        self.pa = pyaudio.PyAudio()
        self.model = self._load_model_once(model_name)
        self.state = AudioCaptureState()
        self.listener: keyboard.Listener | None = None
        self.enter_is_down = threading.Event()
        self.stop_event = threading.Event()
        self.audio_queue: queue.Queue[bytes] = queue.Queue()
        self.text_queue: queue.Queue[str] = queue.Queue()
        self.transcribe_thread = threading.Thread(target=self._transcribe_loop, daemon=True)

    @staticmethod
    def _load_model_once(model_name: str) -> WhisperModel:
        try:
            model = WhisperModel(model_name, device="cuda", compute_type="float16")
            print("[stt] Loaded faster-whisper on GPU.")
            return model
        except Exception:
            model = WhisperModel(model_name, device="cpu", compute_type="int8")
            print("[stt] Loaded faster-whisper on CPU.")
            return model

    @staticmethod
    def _bytes_to_float32_mono(audio_bytes: bytes) -> np.ndarray:
        audio_i16 = np.frombuffer(audio_bytes, dtype=np.int16)
        if audio_i16.size == 0:
            return np.array([], dtype=np.float32)
        return (audio_i16.astype(np.float32) / 32768.0).flatten()

    def start(self) -> None:
        self.transcribe_thread.start()
        self.listener = keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=True,
        )
        self.listener.start()
        print("[stt] Hold Enter to talk. Release Enter to transcribe. Press Esc to exit.")

    def stop(self) -> None:
        self.stop_event.set()
        _ = self._stop_recording()
        if self.listener is not None and self.listener.is_alive():
            self.listener.stop()
        self.audio_queue.put(b"")
        self.transcribe_thread.join(timeout=2.0)
        self.pa.terminate()

    def get_transcript(self, timeout: float = 0.1) -> str | None:
        try:
            return self.text_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _capture_loop(self) -> None:
        assert self.state.stream is not None
        while not self.state.stop_capture.is_set() and not self.stop_event.is_set():
            try:
                chunk = self.state.stream.read(FRAMES_PER_BUFFER, exception_on_overflow=False)
                self.state.frames.append(chunk)
            except OSError as exc:
                print(f"[stt] Audio read error: {exc}")
                break

    def _start_recording(self) -> None:
        with self.state.lock:
            if self.state.is_recording:
                return
            try:
                stream = self.pa.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=SAMPLE_RATE,
                    input=True,
                    frames_per_buffer=FRAMES_PER_BUFFER,
                )
            except OSError as exc:
                print(f"[stt] Could not open mic stream: {exc}")
                return

            self.state.is_recording = True
            self.state.frames = []
            self.state.stop_capture.clear()
            self.state.stream = stream
            self.state.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.state.capture_thread.start()
            print("[stt] Recording...")

    def _stop_recording(self) -> bytes:
        with self.state.lock:
            if not self.state.is_recording:
                return b""

            self.state.is_recording = False
            self.state.stop_capture.set()

            if self.state.capture_thread is not None:
                self.state.capture_thread.join(timeout=2.0)

            if self.state.stream is not None:
                try:
                    self.state.stream.stop_stream()
                    self.state.stream.close()
                except OSError:
                    pass

            self.state.stream = None
            self.state.capture_thread = None
            audio_bytes = b"".join(self.state.frames)
            self.state.frames = []
            return audio_bytes

    def _transcribe_loop(self) -> None:
        while not self.stop_event.is_set():
            try:
                audio_bytes = self.audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.stop_event.is_set():
                break
            if not audio_bytes:
                continue

            waveform = self._bytes_to_float32_mono(audio_bytes)
            if waveform.size == 0:
                continue

            try:
                segments, _info = self.model.transcribe(
                    waveform,
                    language="en",
                    vad_filter=True,
                    beam_size=5,
                    temperature=0.0,
                )
                parts = [segment.text.strip() for segment in segments if segment.text.strip()]
                text = " ".join(parts).strip()
                if text:
                    self.text_queue.put(text)
            except Exception as exc:
                print(f"[stt] Transcription failed: {exc}")

    def _on_press(self, key: object) -> None:
        if key == keyboard.Key.esc:
            print("[stt] Exit requested.")
            self.stop_event.set()
            _ = self._stop_recording()
            if self.listener is not None:
                self.listener.stop()
            return

        if key == keyboard.Key.enter:
            if self.enter_is_down.is_set():
                return
            self.enter_is_down.set()
            threading.Thread(target=self._start_recording_after_tts, daemon=True).start()

    def _start_recording_after_tts(self) -> None:
        if self.before_listen is not None:
            try:
                self.before_listen()
            except Exception as exc:
                print(f"[stt] before_listen hook failed: {exc}")

        if self.stop_event.is_set() or not self.enter_is_down.is_set():
            return
        self._start_recording()

    def _on_release(self, key: object) -> None:
        if key != keyboard.Key.enter:
            return
        if not self.enter_is_down.is_set():
            return
        self.enter_is_down.clear()
        audio_bytes = self._stop_recording()
        if self.after_listen is not None:
            try:
                self.after_listen()
            except Exception as exc:
                print(f"[stt] after_listen hook failed: {exc}")
        if audio_bytes:
            self.audio_queue.put(audio_bytes)


def _is_valid_transcript(text: str) -> bool:
    cleaned = (text or "").strip()
    if len(cleaned) < 3:
        return False
    alpha_count = sum(1 for char in cleaned if char.isalpha())
    return alpha_count >= 2


def _conversation_context(turns: deque[str]) -> str:
    if not turns:
        return ""
    return "\n".join(turns)


def _run_gatekeeper(
    transcriber: PushToTalkSTT,
    memory: RecentMemory,
    tts: KokoroTTSEngine | None,
    preset_recipe: str,
) -> str:
    dish = (preset_recipe or "").strip()
    if dish:
        print(f"[ai_remy] Using preset recipe: {dish}")
    else:
        prompt = "What dish are we making today?"
        print(f"[ai_remy] {prompt}")
        if tts:
            tts.speak(prompt)

        while not transcriber.stop_event.is_set():
            spoken = transcriber.get_transcript(timeout=0.2)
            if spoken is None:
                continue
            if not _is_valid_transcript(spoken):
                print("[ai_remy] Ignored unclear recipe input. Please try again.")
                continue
            dish = spoken.strip()
            print(f"[ai_remy] Recipe captured: {dish}")
            break

    if dish:
        if tts:
            tts.speak("One moment while I put together a recipe.")
        steps = generate_recipe_for_dish(dish)
        dish_summary = summarize_dish_from_recipe(dish, steps).strip() or dish
        memory.set_recipe(dish_summary)
        memory.set_recipe_steps(steps)
        print(f"[ai_remy] Recipe finalized: {dish_summary}")
        if tts:
            tts.speak(f"Great. We are making {dish_summary}. I will guide you step by step.")
    else:
        print("[ai_remy] No recipe set; Remy will still answer questions about the current scene.")

    return dish


def main() -> None:
    repo_root = _laptop_dir.parent
    cooking_vision_dir = repo_root / "cooking-vision"
    app_py = cooking_vision_dir / "app.py"
    default_frame_path = cooking_vision_dir / "frames" / "latest.jpg"

    parser = argparse.ArgumentParser(
        description="Run AI Remy: camera capture + local voice + multimodal Gemini"
    )
    parser.add_argument(
        "--stream",
        default="0",
        help='Camera source: 0 for built-in, or ESP32-CAM URL e.g. "http://<IP>:81/stream"',
    )
    parser.add_argument(
        "--frame-path",
        default=str(default_frame_path),
        help="Path to latest frame (must match where cooking-vision writes)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=0.2,
        help="Seconds between frame-buffer checks",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=8.0,
        help="Seconds between frame captures in cooking-vision",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=25.0,
        help="Scene-change threshold for cooking-vision",
    )
    parser.add_argument(
        "--recipe",
        default="",
        help="Optional pre-set dish (e.g. 'Pasta'). If omitted, gatekeeper asks via mic.",
    )
    parser.add_argument(
        "--stt-model",
        default="base.en",
        help="faster-whisper model name (e.g. base.en, small.en)",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Disable local speaker playback",
    )
    args = parser.parse_args()

    if not app_py.exists():
        print(f"[ai_remy] Not found: {app_py}")
        print("         Make sure cooking-vision/app.py exists in the repo.")
        sys.exit(1)

    frame_path = Path(args.frame_path)

    cmd = [
        sys.executable,
        str(app_py),
        "--stream",
        args.stream,
        "--output",
        str(frame_path.parent),
        "--interval",
        str(args.capture_interval),
        "--threshold",
        str(args.scene_threshold),
    ]

    print("[ai_remy] Starting camera capture (cooking-vision)...")
    proc = subprocess.Popen(
        cmd,
        cwd=str(cooking_vision_dir),
        stdout=sys.stdout,
        stderr=sys.stderr,
    )

    time.sleep(3)
    if proc.poll() is not None:
        print("[ai_remy] Camera process exited early. Check stream URL and camera access.")
        sys.exit(1)

    memory = RecentMemory(max_events=10, max_commentaries=5)
    tts = None if args.no_speak else KokoroTTSEngine()
    conversation_turns: deque[str] = deque(maxlen=10)

    frame_buffer = LatestFrameBuffer(frame_path=frame_path, poll_interval=args.poll)

    def before_listen() -> None:
        if tts:
            tts.hold_after_current_sentence()
            _ = tts.wait_until_current_sentence_done(timeout=15.0)

    def after_listen() -> None:
        if tts:
            tts.resume_playback()

    transcriber = PushToTalkSTT(
        model_name=args.stt_model,
        before_listen=before_listen,
        after_listen=after_listen,
    )

    print("[ai_remy] Starting frame buffer and microphone listener...")
    frame_buffer.start()
    transcriber.start()

    try:
        _ = _run_gatekeeper(transcriber, memory, tts, args.recipe)
        print("[ai_remy] Main loop active. I will guide on frame updates, and you can hold Enter anytime.")

        last_processed_mtime: float | None = None

        def run_turn(user_text: str, image_bytes: bytes, label: str) -> None:
            def on_chunk(chunk: str) -> None:
                if tts:
                    tts.enqueue(chunk)

            events, commentary, should_speak = process_frame_streaming(
                image_bytes,
                memory,
                on_comment_chunk=on_chunk,
                user_prompt=user_text,
                conversation_context=_conversation_context(conversation_turns),
            )

            print(f"[ai_remy] source: {label}")
            print(f"[ai_remy] events: {events}")
            if commentary:
                print(f"[ai_remy] commentary: {commentary}")
            if should_speak:
                print(f"[ai_remy] >>> SAY: {commentary}")

            conversation_turns.append(
                f"User: {user_text}\nRemy: {commentary if commentary else '[no response]'}"
            )

        while not transcriber.stop_event.is_set():
            user_text = transcriber.get_transcript(timeout=0.05)
            if user_text is not None:
                if not _is_valid_transcript(user_text):
                    print("[ai_remy] Ignored unclear speech.")
                else:
                    latest_bytes, latest_mtime = frame_buffer.get_latest_frame()
                    if not latest_bytes:
                        print("[ai_remy] Heard you, but no frame is available yet.")
                    else:
                        print(f"[ai_remy] You: {user_text}")
                        if latest_mtime is not None:
                            print(f"[ai_remy] Using latest frame mtime: {latest_mtime:.3f}")
                        try:
                            run_turn(user_text, latest_bytes, label="user-voice")
                        except Exception as exc:
                            print(f"[ai_remy] Pipeline error (user-voice): {exc}")

            image_bytes, frame_mtime = frame_buffer.get_latest_frame()
            if not image_bytes:
                time.sleep(0.1)
                continue
            if frame_mtime is None:
                time.sleep(0.1)
                continue
            if last_processed_mtime is not None and frame_mtime <= last_processed_mtime:
                time.sleep(0.1)
                continue
            last_processed_mtime = frame_mtime

            auto_prompt = "Please guide me through the next recipe step based on this frame."
            print(f"[ai_remy] Processing frame update at mtime: {frame_mtime:.3f}")
            try:
                run_turn(auto_prompt, image_bytes, label="frame-update")
            except Exception as exc:
                print(f"[ai_remy] Pipeline error (frame-update): {exc}")

    except KeyboardInterrupt:
        print("\n[ai_remy] Shutting down...")
    finally:
        transcriber.stop()
        frame_buffer.stop()
        if tts:
            tts.close()
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

        frames_dir = cooking_vision_dir / "frames"
        if frames_dir.exists():
            try:
                shutil.rmtree(frames_dir)
                print(f"[ai_remy] Deleted frames folder: {frames_dir}")
            except OSError as exc:
                print(f"[ai_remy] Could not delete frames folder {frames_dir}: {exc}")

        print("[ai_remy] Done.")


if __name__ == "__main__":
    main()
