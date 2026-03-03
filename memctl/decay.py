"""Memory decay engine for memctl."""

import math
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

CONFIG_PATH = Path.home() / ".memctl" / "config.toml"
DB_PATH = Path.home() / ".memctl" / "memory.db"


def _load_config() -> dict:
    """Load decay config from TOML, or return defaults."""
    defaults = {
        "enabled": True,
        "curve": "exponential",
        "half_life_days": 30,
        "min_score": 0.1,
        "boost_on_access": 0.3,
    }
    if not CONFIG_PATH.exists():
        return defaults

    try:
        import tomllib  # Python 3.11+
    except ImportError:
        try:
            import tomli as tomllib  # pip install tomli for older Python
        except ImportError:
            return defaults  # fallback to defaults if no TOML parser

    try:
        with open(CONFIG_PATH, "rb") as f:
            data = tomllib.load(f)
        cfg = data.get("decay", {})
        return {**defaults, **cfg}
    except Exception:
        return defaults


def _compute_decay_score(
    current_score: float,
    last_accessed: str,
    curve: str,
    half_life_days: int,
) -> float:
    """Compute new decay score based on time since last access."""
    try:
        last = datetime.fromisoformat(last_accessed.replace("Z", "+00:00"))
    except ValueError:
        last = datetime.now(timezone.utc)

    days_since = (datetime.now(timezone.utc) - last).total_seconds() / 86400

    if curve == "exponential":
        # S(t) = S0 * 2^(-t / half_life)
        new_score = current_score * math.pow(2, -days_since / max(half_life_days, 1))
    elif curve == "linear":
        # S(t) = S0 - (S0 / half_life) * t
        decay_rate = current_score / max(half_life_days, 1)
        new_score = current_score - decay_rate * days_since
    elif curve == "step":
        # S(t) = S0 if t < half_life else 0
        new_score = current_score if days_since < half_life_days else 0.0
    else:
        new_score = current_score  # unknown curve = no decay

    return max(0.0, min(1.0, new_score))


def run_decay(
    dry_run: bool = True,
    db_path: Optional[Path] = None,
) -> Tuple[List[dict], List[dict]]:
    """
    Run the decay engine.
    Returns (to_delete, to_update) lists.
    - to_delete: memories whose new score < min_score
    - to_update: memories whose score changed but are above min_score
    """
    cfg = _load_config()
    if not cfg["enabled"]:
        return [], []

    path = db_path or DB_PATH
    if not path.exists():
        return [], []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT id, agent, content, decay_score, last_accessed FROM memories"
    ).fetchall()

    to_delete = []
    to_update = []

    for row in rows:
        new_score = _compute_decay_score(
            row["decay_score"],
            row["last_accessed"],
            cfg["curve"],
            cfg["half_life_days"],
        )

        # Only update if score actually changed
        if abs(new_score - row["decay_score"]) < 0.001:
            continue

        entry = {
            "id": row["id"],
            "agent": row["agent"],
            "content": row["content"][:80],
            "old_score": round(row["decay_score"], 3),
            "new_score": round(new_score, 3),
            "last_accessed": row["last_accessed"][:16].replace("T", " "),
        }

        if new_score < cfg["min_score"]:
            to_delete.append(entry)
        else:
            to_update.append(entry)

    if not dry_run:
        # Apply score updates
        for entry in to_update:
            conn.execute(
                "UPDATE memories SET decay_score = ? WHERE id = ?",
                (entry["new_score"], entry["id"]),
            )
        # Delete low-score memories
        for entry in to_delete:
            conn.execute("DELETE FROM memories WHERE id = ?", (entry["id"],))

        conn.commit()

    conn.close()
    return to_delete, to_update


def write_default_config() -> Path:
    """Write a default config.toml if none exists."""
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not CONFIG_PATH.exists():
        CONFIG_PATH.write_text(
            """[decay]
enabled = true
curve = "exponential"   # exponential | linear | step
half_life_days = 30     # 50% score loss per 30 days without access
min_score = 0.1         # below this = deletion candidate
boost_on_access = 0.3   # score boost on each recall

[consolidation]
auto = false
threshold = 0.85        # cosine similarity > 0.85 = merge candidates
"""
        )
    return CONFIG_PATH
