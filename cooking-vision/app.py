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
from urllib.parse import urlparse
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

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
RECONNECT_DELAY_S   = 2.0  # delay before reconnect attempt after stream failure


# ──────────────────────────────────────────────
# SCENE CHANGE DETECTION
# ──────────────────────────────────────────────

def scene_changed(prev_gray: np.ndarray | None, curr_gray: np.ndarray, threshold: float) -> bool:
    """Return True if the mean absolute pixel difference exceeds the threshold."""
    if prev_gray is None:
        return True
    diff = np.mean(np.abs(curr_gray.astype(np.float32) - prev_gray.astype(np.float32)))
    return diff > threshold


def normalize_stream_source(source: int | str) -> int | str:
    """Normalize ESP32-CAM host URLs to the MJPEG endpoint expected by OpenCV."""
    if isinstance(source, int):
        return source

    candidate = str(source).strip()
    parsed = urlparse(candidate)

    if parsed.scheme in ("http", "https"):
        path = parsed.path or ""
        if path in ("", "/"):
            host = parsed.hostname or ""
            if not host:
                return candidate
            return f"{parsed.scheme}://{host}:81/stream"

    return candidate


def is_http_source(source: int | str) -> bool:
    return isinstance(source, str) and source.startswith(("http://", "https://"))


def stream_url_to_snapshot_url(stream_url: str) -> str:
    parsed = urlparse(stream_url)
    host = parsed.hostname or ""
    if not host:
        return stream_url
    return f"{parsed.scheme}://{host}/capture"


def resize_preserve_aspect(frame: np.ndarray) -> np.ndarray:
    """Resize to RESIZE_MAX_W while preserving aspect ratio."""
    h, w = frame.shape[:2]
    scale = RESIZE_MAX_W / w
    return cv2.resize(frame, (RESIZE_MAX_W, int(h * scale)))


def fetch_snapshot_frame(snapshot_url: str, timeout_s: float = 5.0) -> np.ndarray | None:
    """Fetch one JPEG snapshot frame from ESP32-CAM /capture endpoint."""
    try:
        with urlopen(snapshot_url, timeout=timeout_s) as response:
            data = response.read()
        if not data:
            return None
        array = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        return frame
    except (URLError, HTTPError, TimeoutError, OSError):
        return None


def open_stream_with_retry(
    stream_url: int | str,
    max_attempts: int | None = None,
) -> tuple[cv2.VideoCapture | None, np.ndarray | None]:
    """Try to open camera stream and return first valid frame; optionally stop after max_attempts."""
    attempts = 0
    while True:
        attempts += 1
        print(f"[camera] Connecting to: {stream_url}")
        cap = cv2.VideoCapture(stream_url)
        if not cap.isOpened():
            print(f"[camera] Cannot open stream yet. Retrying in {RECONNECT_DELAY_S:.1f}s…")
            cap.release()
            if max_attempts is not None and attempts >= max_attempts:
                return None, None
            time.sleep(RECONNECT_DELAY_S)
            continue

        for _ in range(5):
            cap.read()

        ret, test_frame = cap.read()
        if not ret or test_frame is None:
            print(f"[camera] Opened stream but got no frames. Retrying in {RECONNECT_DELAY_S:.1f}s…")
            cap.release()
            if max_attempts is not None and attempts >= max_attempts:
                return None, None
            time.sleep(RECONNECT_DELAY_S)
            continue

        return cap, test_frame


# ──────────────────────────────────────────────
# MAIN LOOP
# ──────────────────────────────────────────────

def run(stream_url: int | str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    abs_output = os.path.abspath(output_dir)

    stream_url = normalize_stream_source(stream_url)

    use_snapshot_fallback = False
    snapshot_url = None

    if is_http_source(stream_url):
        cap, test_frame = open_stream_with_retry(stream_url, max_attempts=3)
        if cap is None or test_frame is None:
            use_snapshot_fallback = True
            snapshot_url = stream_url_to_snapshot_url(stream_url)
            print(f"[camera] MJPEG stream unavailable. Falling back to snapshots: {snapshot_url}")
            while True:
                test_frame = fetch_snapshot_frame(snapshot_url)
                if test_frame is not None:
                    break
                print(f"[camera] Snapshot fetch failed. Retrying in {RECONNECT_DELAY_S:.1f}s…")
                time.sleep(RECONNECT_DELAY_S)
    else:
        cap, test_frame = open_stream_with_retry(stream_url)

    if test_frame is None:
        raise RuntimeError("No frame received from camera source")

    print(f"[camera] Stream opened ✓  (resolution: {test_frame.shape[1]}×{test_frame.shape[0]})")
    print(f"[frames] Saving frames to: {abs_output}")
    print("[loop]   Press Q in the preview window or Ctrl+C to quit.\n")

    # Save one frame immediately so downstream watcher has input right away
    first_frame = resize_preserve_aspect(test_frame)
    first_filename = os.path.join(output_dir, FRAME_FILENAME)
    cv2.imwrite(first_filename, first_frame)
    print(f"[saved]  {first_filename} (initial)")

    prev_gray     = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    last_grab     = 0.0
    saved_count   = 1
    skipped_count = 0

    try:
        while True:
            if use_snapshot_fallback:
                frame = fetch_snapshot_frame(snapshot_url)
                if frame is None:
                    print(f"[camera] Snapshot fetch failed — retrying in {RECONNECT_DELAY_S:.1f}s…")
                    time.sleep(RECONNECT_DELAY_S)
                    continue
            else:
                ret, frame = cap.read()
                if not ret:
                    print("[camera] Frame read failed — reconnecting stream…")
                    cap.release()
                    max_attempts = 3 if is_http_source(stream_url) else None
                    cap, recovered = open_stream_with_retry(stream_url, max_attempts=max_attempts)
                    if cap is None or recovered is None:
                        if is_http_source(stream_url):
                            use_snapshot_fallback = True
                            snapshot_url = stream_url_to_snapshot_url(stream_url)
                            print(f"[camera] Switched to snapshot fallback: {snapshot_url}")
                            continue
                        time.sleep(RECONNECT_DELAY_S)
                        continue
                    frame = recovered
                    print("[camera] Stream reconnected ✓")
                    continue

            # Resize to max width while preserving aspect ratio
            frame = resize_preserve_aspect(frame)
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
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print(f"[done] Latest frame kept at: {os.path.join(abs_output, FRAME_FILENAME)}")


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
