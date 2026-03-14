# GenAI Ratatouille — Cooking Vision 🐀🍳

Real-time cooking assistant powered by an **ESP32-CAM**, **OpenCV**, and **Gemini 1.5 Flash**, with spoken feedback via **piper-tts**.

## Architecture

```
ESP32-CAM  (MJPEG stream)
     ↓
OpenCV     (frame capture + scene-change filter)
     ↓
Gemini 1.5 Flash  (cooking action recognition + commentary)
     ↓
piper-tts  (spoken output)
     ↓
Speaker
```

## Setup

### 1. Python environment

```bash
cd cooking-vision
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Gemini API key

Get a key from [Google AI Studio](https://aistudio.google.com/), then:

```bash
export GEMINI_API_KEY="your-key-here"
```

### 3. piper-tts (optional, for voice output)

```bash
# macOS
brew install piper
# Download a voice model
piper --download-voice en_US-lessac-medium
```

> Without piper, commentary is printed to the terminal instead.

### 4. ESP32-CAM

Flash your ESP32-CAM with the `CameraWebServer` Arduino example.  
Update the stream URL in `app.py` or pass it via `--stream`:

```
DEFAULT_STREAM_URL = "http://<your-esp32-ip>:81/stream"
```

## Running

```bash
# Basic run
python app.py

# Custom stream URL
python app.py --stream "http://192.168.1.42:81/stream"

# Save the session to a video file
python app.py --save-video runs/session.mp4

# Slower API calls (every 8 seconds) + more sensitive scene detection
python app.py --interval 8 --scene-threshold 15
```

## Tuning

| Flag | Default | What it does |
|---|---|---|
| `--interval` | `4` | Seconds between Gemini calls |
| `--scene-threshold` | `25` | Mean pixel diff to trigger a new call. Lower = more sensitive |

**Tip:** A threshold of `25` skips 80–95% of redundant frames automatically. Drop it to `10` if Gemini is missing quick actions.

## File Structure

```
cooking-vision/
  app.py            ← main pipeline script
  requirements.txt
  README.md
```

## Roadmap

- [ ] WebSocket broadcast of Gemini commentary (for teammate feedback overlay)
- [ ] Recipe context injection (tell Gemini which recipe you're following)
- [ ] Multi-camera angle support
- [ ] Save annotated highlight clips of key cooking moments
