"""Load config from environment (API key, image size, quality)."""
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from laptop/ or project root
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
MAX_IMAGE_PX = int(os.getenv("MAX_IMAGE_PX", "1024"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))
