"""
AI Remy — Watch cooking-vision frames and run Gemini pipeline in real time.

Poll frames/latest.jpg (written by cooking-vision/app.py every ~5s when the scene
changes). Each time the file is updated, load it and call the Gemini pipeline;
print (and optionally speak) commentary when something meaningful is detected.

Usage:
  # Terminal 1: run the camera capture (writes to cooking-vision/frames/latest.jpg)
  cd cooking-vision && python app.py

  # Terminal 2: run Remy (watches that file and calls Gemini when it changes)
    cd laptop && .\\venv\\Scripts\\Activate.ps1 && python run_remy.py

  # Or set FRAME_PATH in .env to point at your frame file.
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure laptop/ is on path so ai_remy can be imported
_laptop_dir = Path(__file__).resolve().parent
if str(_laptop_dir) not in sys.path:
    sys.path.insert(0, str(_laptop_dir))

from ai_remy import config
from ai_remy.pipeline import process_frame_streaming
from ai_remy.state.memory import RecentMemory
from ai_remy.tts_engine import KokoroTTSEngine


POLL_INTERVAL_S = 1.0  # check file mtime every second


def run(frame_path: str, poll_interval: float = POLL_INTERVAL_S, speak: bool = True) -> None:
    path = Path(frame_path)
    if not path.parent.exists():
        print(f"[remy] Frame directory does not exist: {path.parent}")
        print("       Run cooking-vision first: cd cooking-vision && python app.py")
        sys.exit(1)

    if not config.GEMINI_API_KEY:
        print("[remy] GEMINI_API_KEY not set. Add it to laptop/.env")
        sys.exit(1)

    memory = RecentMemory(max_events=10, max_commentaries=5)
    tts = KokoroTTSEngine() if speak else None
    last_mtime: float | None = None
    print(f"[remy] Watching: {path.resolve()}")
    print("[remy] Each time cooking-vision saves a new frame, Gemini will analyze it.")
    print("[remy] Press Ctrl+C to quit.\n")

    try:
        while True:
            if not path.exists():
                time.sleep(poll_interval)
                continue

            mtime = path.stat().st_mtime
            if last_mtime is not None and mtime <= last_mtime:
                time.sleep(poll_interval)
                continue

            last_mtime = mtime
            try:
                image_bytes = path.read_bytes()
            except OSError as e:
                print(f"[remy] Could not read {path}: {e}")
                time.sleep(poll_interval)
                continue

            if len(image_bytes) == 0:
                time.sleep(poll_interval)
                continue

            def on_chunk(chunk: str) -> None:
                if tts:
                    tts.enqueue(chunk)

            try:
                events, commentary, should_speak = process_frame_streaming(
                    image_bytes,
                    memory,
                    on_comment_chunk=on_chunk,
                )
            except Exception as e:
                print(f"[remy] Pipeline error: {e}")
                time.sleep(poll_interval)
                continue

            print(f"[remy] events: {events}")
            if commentary:
                print(f"[remy] commentary: {commentary}")
            if should_speak:
                print(f"[remy] >>> SAY: {commentary}")

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        print("\n[remy] Done.")
    finally:
        if tts:
            tts.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Watch cooking-vision frame file and run Gemini pipeline")
    parser.add_argument(
        "--frame-path",
        default=config.FRAME_PATH,
        help=f"Path to latest frame (default: {config.FRAME_PATH})",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=POLL_INTERVAL_S,
        help="Seconds between file checks",
    )
    parser.add_argument(
        "--no-speak",
        action="store_true",
        help="Disable local speaker playback",
    )
    args = parser.parse_args()
    run(frame_path=args.frame_path, poll_interval=args.poll, speak=not args.no_speak)
