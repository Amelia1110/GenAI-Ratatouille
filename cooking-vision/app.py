"""
GenAI Ratatouille — Frame Capture
===================================
Connects to the ESP32-CAM MJPEG stream via OpenCV,
grabs a frame every N seconds, skips frames where the
scene hasn't changed, and saves them to an output folder
for the Gemini integration step.
"""

import os
import time
import argparse

import cv2
import numpy as np


# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────

DEFAULT_STREAM_URL  = 0     # 0 = built-in MacBook camera; swap for "http://<IP>:81/stream" when using ESP32-CAM
FRAME_INTERVAL_S    = 15     # seconds between captured frames
SCENE_CHANGE_THRESH = 25    # mean pixel diff (0-255); lower = more sensitive
RESIZE_MAX_W        = 640    # resize to this width, height scales to preserve aspect ratio
OUTPUT_DIR          = "frames"  # folder where frames are saved
FRAME_FILENAME      = "latest.jpg"  # single output frame, overwritten each save


# ──────────────────────────────────────────────
# SCENE CHANGE DETECTION
# ──────────────────────────────────────────────

def scene_changed(prev_gray: np.ndarray | None, curr_gray: np.ndarray, threshold: float) -> bool:
    """Return True if the mean absolute pixel difference exceeds the threshold."""
    if prev_gray is None:
        return True
    diff = np.mean(np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32)))
    return diff > threshold


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def run(stream_url: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    abs_output = os.path.abspath(output_dir)

    print(f"[camera] Connecting to: {stream_url}")
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open stream: {stream_url}")

    # Warm up the camera — macOS needs a moment before frames come through
    for _ in range(5):
        cap.read()

    # Verify we're actually getting frames
    ret, test_frame = cap.read()
    if not ret or test_frame is None:
        raise RuntimeError(
            "Camera opened but returned no frames.\n"
            "On macOS: go to System Settings → Privacy & Security → Camera\n"
            "and grant access to Terminal (or your IDE)."
        )
    print(f"[camera] Stream opened ✓  (resolution: {test_frame.shape[1]}×{test_frame.shape[0]})")
    print(f"[frames] Saving frames to: {abs_output}")
    print("[loop]   Press Q in the preview window or Ctrl+C to quit.\n")

    prev_gray     = None
    last_grab     = 0.0
    saved_count   = 0
    skipped_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[camera] Frame read failed — retrying…")
                time.sleep(0.5)
                continue

            # Resize to max width while preserving aspect ratio
            h, w  = frame.shape[:2]
            scale = RESIZE_MAX_W / w
            frame = cv2.resize(frame, (RESIZE_MAX_W, int(h * scale)))
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Show live preview window
            cv2.imshow("Cooking Vision — press Q to quit", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            now = time.time()
            if (now - last_grab) >= FRAME_INTERVAL_S:
                if scene_changed(prev_gray, curr_gray, SCENE_CHANGE_THRESH):
                    filename = os.path.join(output_dir, FRAME_FILENAME)
                    cv2.imwrite(filename, frame)
                    saved_count += 1
                    print(f"[saved]  {filename}")
                    prev_gray = curr_gray
                else:
                    skipped_count += 1
                    print(f"[skip]   No change detected (skips: {skipped_count})")

                last_grab = now

    except KeyboardInterrupt:
        print(f"\n[done] Saved: {saved_count} frames | Skipped: {skipped_count}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        # Clean up all saved frames on exit
        removed = [f for f in os.listdir(output_dir) if f.endswith(".jpg")]
        for f in removed:
            os.remove(os.path.join(output_dir, f))
        if removed:
            print(f"[cleanup] Deleted {len(removed)} frames from {abs_output}")


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESP32-CAM frame capture for GenAI Ratatouille")
    parser.add_argument("--stream",    default=DEFAULT_STREAM_URL,
                        help="Camera source: device index (0, 1…) or ESP32-CAM URL e.g. http://<IP>:81/stream")
    parser.add_argument("--output",    default=OUTPUT_DIR,           help="Output folder for frames")
    parser.add_argument("--interval",  type=float, default=FRAME_INTERVAL_S,    help="Seconds between frames")
    parser.add_argument("--threshold", type=float, default=SCENE_CHANGE_THRESH, help="Scene-change sensitivity")
    args = parser.parse_args()

    FRAME_INTERVAL_S    = args.interval
    SCENE_CHANGE_THRESH = args.threshold

    # Allow passing a bare integer (device index) or a URL string
    source = int(args.stream) if str(args.stream).isdigit() else args.stream
    run(stream_url=source, output_dir=args.output)
