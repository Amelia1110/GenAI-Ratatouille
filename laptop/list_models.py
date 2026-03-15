"""
List Gemini models available for your API key.
Run with venv active:  python list_models.py

Use this to see which model names work (e.g. for GEMINI_MODEL in .env).
"""

import sys
from pathlib import Path

_laptop_dir = Path(__file__).resolve().parent
if str(_laptop_dir) not in sys.path:
    sys.path.insert(0, str(_laptop_dir))

from ai_remy import config
import google.generativeai as genai

if not config.GEMINI_API_KEY:
    print("GEMINI_API_KEY not set in .env")
    sys.exit(1)

genai.configure(api_key=config.GEMINI_API_KEY)

print("Models that support generateContent:\n")
for m in genai.list_models():
    methods = m.supported_generation_methods or []
    if "generateContent" in methods:
        # Model name is often "models/gemini-2.5-flash" etc.
        print(m.name)
print("\nUse one of these as GEMINI_MODEL in .env (e.g. gemini-2.5-flash).")
