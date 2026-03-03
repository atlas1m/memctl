"""LLM-powered memory consolidation for memctl."""

import os
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple

DB_PATH = Path.home() / ".memctl" / "memory.db"
DEFAULT_THRESHOLD = 0.85
DEFAULT_MIN_CLUSTER = 2


def _load_consolidation_config() -> dict:
    """Load consolidation config from TOML or return defaults."""
    defaults = {
        "threshold": DEFAULT_THRESHOLD,
        "min_memories": DEFAULT_MIN_CLUSTER,
        "auto": False,
    }
    config_path = Path.home() / ".memctl" / "config.toml"
    if not config_path.exists():
        return defaults
    try:
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
        cfg = data.get("consolidation", {})
        return {**defaults, **cfg}
    except Exception:
        return defaults


def _find_clusters(
    rows: List[sqlite3.Row],
    threshold: float,
) -> List[List[sqlite3.Row]]:
    """Group memories by cosine similarity above threshold."""
    from memctl.embeddings import unpack_embedding, cosine_similarity

    # Only consider memories with embeddings
    with_emb = [(r, unpack_embedding(r["embedding"])) for r in rows if r["embedding"]]
    if len(with_emb) < 2:
        return []

    visited = set()
    clusters = []

    for i, (row_i, vec_i) in enumerate(with_emb):
        if i in visited:
            continue
        cluster = [row_i]
        visited.add(i)
        for j, (row_j, vec_j) in enumerate(with_emb):
            if j == i or j in visited:
                continue
            sim = cosine_similarity(vec_i, vec_j)
            if sim >= threshold:
                cluster.append(row_j)
                visited.add(j)
        if len(cluster) >= 2:
            clusters.append(cluster)

    return clusters


def _merge_with_llm(memories: List[str], agent: str) -> Optional[str]:
    """Use Claude Haiku to merge a cluster of similar memories into one."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        # Fallback: simple concatenation with dedup
        seen = set()
        merged = []
        for m in memories:
            if m not in seen:
                seen.add(m)
                merged.append(m)
        return " | ".join(merged)

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        memories_text = "\n".join(f"- {m}" for m in memories)
        prompt = f"""You are a memory consolidation assistant.
Merge these {len(memories)} related memories about the same topic into ONE concise, accurate memory.
Keep all important facts. Remove redundancy. Be specific. Max 200 chars.

Memories to merge:
{memories_text}

Return ONLY the merged memory text, nothing else."""

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        # Fallback on API error
        return f"[Consolidated] {memories[0]} (+{len(memories)-1} similar)"


def consolidate(
    agent: Optional[str] = None,
    dry_run: bool = True,
    threshold: Optional[float] = None,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Find and merge similar memories.
    Returns list of consolidation actions performed (or planned in dry-run).
    """
    cfg = _load_consolidation_config()
    th = threshold or cfg["threshold"]

    path = db_path or DB_PATH
    if not path.exists():
        return []

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    if agent:
        rows = conn.execute(
            "SELECT * FROM memories WHERE agent = ? AND embedding IS NOT NULL",
            (agent,),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM memories WHERE embedding IS NOT NULL"
        ).fetchall()

    clusters = _find_clusters(rows, th)
    actions = []

    for cluster in clusters:
        memories_text = [r["content"] for r in cluster]
        ids_to_merge = [r["id"] for r in cluster]
        agent_name = cluster[0]["agent"]

        merged = _merge_with_llm(memories_text, agent_name)

        action = {
            "agent": agent_name,
            "original_count": len(cluster),
            "original_ids": ids_to_merge,
            "original_texts": memories_text,
            "merged": merged,
        }
        actions.append(action)

        if not dry_run and merged:
            # Delete originals
            placeholders = ",".join("?" * len(ids_to_merge))
            conn.execute(
                f"DELETE FROM memories WHERE id IN ({placeholders})",
                ids_to_merge,
            )

            # Store merged memory
            from memctl.db import store_memory
            avg_importance = sum(r["importance"] for r in cluster) / len(cluster)
            tags_combined = list(set(
                t for r in cluster
                for t in (r["tags"].split(",") if r["tags"] else [])
                if t
            ))
            store_memory(
                merged,
                agent=agent_name,
                tags=tags_combined,
                importance=avg_importance,
                db_path=path,
            )

            conn.commit()

    conn.close()
    return actions
