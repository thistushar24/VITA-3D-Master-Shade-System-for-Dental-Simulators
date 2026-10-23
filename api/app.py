"""Vercel entrypoint (Python function). Re-exports the FastAPI app that lives in backend/dental_reg.

Vercel runs from the project root and bundles reachable files; model assets are read from data/demo_library.
"""
import sys
from pathlib import Path

# Add backend directory to sys.path (parent of api/ is the project root)
root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root / "backend"))

from dental_reg.app import app  # noqa: E402,F401
