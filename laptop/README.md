# AI Remy — Gemini pipeline + cooking-vision integration

Gemini analyzes each new frame produced by **cooking-vision** in real time and prints (or speaks) Remy-style commentary.

## Pipeline

1. **cooking-vision/app.py** captures from the camera every 5 seconds and saves a frame to `cooking-vision/frames/latest.jpg` when the scene changes.
2. **run_remy.py** watches that file; when it is updated, loads the image and sends it to the Gemini pipeline, then prints the commentary.

## Setup

1. In `laptop/`: create venv and install deps (see below if not done).
2. Copy `laptop/.env.example` to `laptop/.env` and set `GEMINI_API_KEY`.
3. Install cooking-vision deps: `cd cooking-vision && pip install -r requirements.txt` (or use its own venv).

## Run the full pipeline

**One command (recommended):** camera + Gemini in a single process. From `laptop/` with venv active:
```bash
cd laptop
.\venv\Scripts\Activate.ps1
python run_ai_remy.py
```
For ESP32-CAM: `python run_ai_remy.py --stream "http://<IP>:81/stream"`. Press **Ctrl+C** to stop both camera and Gemini.

**Two terminals (alternative):** run camera and watcher separately:
- Terminal 1: `cd cooking-vision && python app.py`
- Terminal 2: `cd laptop && .\venv\Scripts\Activate.ps1 && python run_remy.py`

Each time a new frame is saved, Remy analyzes it and prints events and commentary. When the comment is meaningful and not a repeat, it prints `>>> SAY: ...` (ready for TTS later).

**Override frame path:** `python run_ai_remy.py --frame-path path/to/frames/latest.jpg` or set `FRAME_PATH` in `.env`.

### Vocal session (no typing)

Everything is voice-based. When you run the app:

1. **Remy asks out loud:** “What dish are we making today?”
2. **You answer via mic:** e.g. “Pasta”, “Scrambled eggs”.
3. **Remy generates a recipe** for that dish and saves it in memory for the session.
4. **Remy says:** “Got it. I’ve got a recipe for [dish]. Let’s get started. I’ll watch and guide you as you cook.”
5. **While you cook**, Remy watches the camera and guides you through the recipe with spoken comments and next-step suggestions.

You don’t type anything. Mic input uses `SpeechRecognition` (e.g. `pip install SpeechRecognition pyaudio`). To pre-set the dish without voice (e.g. for testing), set `RECIPE=Pasta` in `.env` or run with `--recipe "Pasta"`.

## Programmatic use

```python
from ai_remy.pipeline import process_frame
from ai_remy.state.memory import RecentMemory

memory = RecentMemory(max_events=10, max_commentaries=5)
memory.set_recipe("pasta carbonara")  # optional: for mentor-style guidance
image_bytes = Path("cooking-vision/frames/latest.jpg").read_bytes()
events, commentary, should_speak = process_frame(image_bytes, memory)
if should_speak:
    print(commentary)  # or send to TTS
```

## Design

See [../docs/AI_REMY_DESIGN.md](../docs/AI_REMY_DESIGN.md) and [../docs/PROMPTS_EXAMPLES.md](../docs/PROMPTS_EXAMPLES.md).
