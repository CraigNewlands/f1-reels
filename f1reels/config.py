import os
from pathlib import Path

CACHE_DIR = Path(os.getenv("F1_CACHE_DIR", Path.home() / ".fastf1_cache"))
OUTPUT_DIR = Path(os.getenv("F1_OUTPUT_DIR", "output"))
DEFAULT_FPS = int(os.getenv("F1_FPS", "30"))
DEFAULT_DURATION = int(os.getenv("F1_DURATION", "45"))
