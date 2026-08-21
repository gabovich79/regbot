import os
import sys
from pathlib import Path

# Make application modules importable whether pytest is invoked from the repo
# directly or by an external verification harness.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# Services construct API clients at import time; tests replace those clients.
os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("GOOGLE_API_KEY", "test-google-key")
