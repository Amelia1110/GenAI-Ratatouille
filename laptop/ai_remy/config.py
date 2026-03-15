"""Load config from environment (API key, image size, quality)."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from laptop/ or project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")  # e.g. gemini-2.5-flash, gemini-2.5-flash-lite
MAX_IMAGE_PX = int(os.getenv("MAX_IMAGE_PX", "1024"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

# Path to the frame file written by cooking-vision (default: cooking-vision/frames/latest.jpg from repo root)
_default_frame = Path(__file__).resolve().parent.parent.parent / "cooking-vision" / "frames" / "latest.jpg"
FRAME_PATH = os.getenv("FRAME_PATH", str(_default_frame))
