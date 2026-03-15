# AI Remy — Gemini pipeline (this branch only)

This branch contains **only the Gemini implementation**: image preprocessing, scene analysis, event extraction, commentary generation, and filtering. No OpenCV or video capture — you will pull the ingestion code from another branch and pass frames into this pipeline.

## Setup

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   venv\Scripts\activate   # Windows
   pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` (and optionally `MAX_IMAGE_PX`, `JPEG_QUALITY`).
3. Use the pipeline from your ingestion branch:
   ```python
   from ai_remy.state.memory import RecentMemory
   from ai_remy.pipeline import process_frame

   memory = RecentMemory(max_events=10, max_commentaries=5)
   # frame = your_capture.read_frame()  # from the other branch
   events, commentary, should_speak = process_frame(frame, memory)
   if should_speak:
       your_tts.speak(commentary)
   ```

## Design

See [../docs/AI_REMY_DESIGN.md](../docs/AI_REMY_DESIGN.md) for architecture and pipeline. Prompt examples: [../docs/PROMPTS_EXAMPLES.md](../docs/PROMPTS_EXAMPLES.md).
