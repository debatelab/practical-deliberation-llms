# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Ensure the project root (which contains `experiments/`) is on sys.path
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
