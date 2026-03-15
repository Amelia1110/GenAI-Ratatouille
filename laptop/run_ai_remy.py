"""
AI Remy — Single entry point: camera capture + Gemini commentary in one process.

Starts cooking-vision (camera, saves frames to cooking-vision/frames/latest.jpg)
in a subprocess, then runs the Gemini watcher in the main process. One script,
one terminal. Press Ctrl+C to stop both.

Usage:
  cd laptop
    .\\venv\\Scripts\\Activate.ps1
  python run_ai_remy.py

  # ESP32-CAM stream:
  python run_ai_remy.py --stream "http://192.168.1.42:81/stream"
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure laptop/ is on path for ai_remy
_laptop_dir = Path(__file__).resolve().parent
if str(_laptop_dir) not in sys.path:
    sys.path.insert(0, str(_laptop_dir))

from run_remy import run as run_watcher


def main() -> None:
    repo_root = _laptop_dir.parent
    cooking_vision_dir = repo_root / "cooking-vision"
    app_py = cooking_vision_dir / "app.py"
    frame_path = cooking_vision_dir / "frames" / "latest.jpg"

    parser = argparse.ArgumentParser(
        description="Run AI Remy: camera capture + Gemini commentary (single process)"
    )
    parser.add_argument(
        "--stream",
        default="0",
        help='Camera source: 0 for built-in, or ESP32-CAM URL e.g. "http://<IP>:81/stream"',
    )
    parser.add_argument(
        "--frame-path",
        default=str(frame_path),
        help="Path to latest frame (must match where cooking-vision writes)",
    )
    parser.add_argument(
        "--poll",
        type=float,
        default=1.0,
        help="Seconds between frame file checks",
    )
    parser.add_argument(
        "--capture-interval",
        type=float,
        default=5.0,
        help="Seconds between frame captures in cooking-vision",
    )
    parser.add_argument(
        "--scene-threshold",
        type=float,
        default=25.0,
        help="Scene-change threshold for cooking-vision",
    )
    args = parser.parse_args()

    if not app_py.exists():
        print(f"[ai_remy] Not found: {app_py}")
        print("         Make sure cooking-vision/app.py exists in the repo.")
        sys.exit(1)

    # Start cooking-vision in subprocess (writes to cooking_vision_dir/frames/latest.jpg)
    stream_arg = args.stream
    cmd = [
        sys.executable,
        str(app_py),
        "--stream",
        stream_arg,
        "--output",
        str(Path(args.frame_path).parent),
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

    # Give camera time to open and optionally write first frame
    time.sleep(3)

    if proc.poll() is not None:
        print("[ai_remy] Camera process exited early. Check stream URL and camera access.")
        sys.exit(1)

    print("[ai_remy] Starting Gemini watcher...")
    try:
        run_watcher(frame_path=args.frame_path, poll_interval=args.poll)
    except KeyboardInterrupt:
        print("\n[ai_remy] Shutting down...")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("[ai_remy] Done.")


if __name__ == "__main__":
    main()
