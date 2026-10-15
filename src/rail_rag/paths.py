"""Filesystem locations. Override the data directory with RAIL_RAG_DATA_DIR."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("RAIL_RAG_DATA_DIR", REPO_ROOT / "data"))
MANIFEST = DATA_DIR / "sources.yaml"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
