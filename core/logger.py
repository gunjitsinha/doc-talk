"""Simple JSONL logger for storing QA interactions.

Appends one JSON object per line to `logs/log.jsonl`.
Each record contains: timestamp, question, answer, retrieved_chunks, llm_model, latency_ms
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "log.jsonl"


def ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log_interaction(entry: Dict[str, Any]) -> None:
    """Append a JSON line to the log file.

    Args:
        entry: Dictionary serializable to JSON
    """
    ensure_log_dir()
    # Ensure timestamp is present
    if "timestamp" not in entry:
        entry["timestamp"] = datetime.utcnow().isoformat() + "Z"

    # Append as a single JSON line
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

