"""SQLite database layer for memctl."""

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional


DB_PATH = Path.home() / ".memctl" / "memory.db"


def get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """Return a SQLite connection, creating the DB if needed."""
    path = db_path or DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create tables if they don't exist yet."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id          TEXT PRIMARY KEY,
            agent       TEXT NOT NULL DEFAULT 'default',
            content     TEXT NOT NULL,
            tags        TEXT NOT NULL DEFAULT '',
            importance  REAL NOT NULL DEFAULT 0.5,
            created_at  TEXT NOT NULL,
            last_accessed TEXT NOT NULL,
            access_count  INTEGER NOT NULL DEFAULT 0,
            decay_score   REAL NOT NULL DEFAULT 1.0
        );

        CREATE INDEX IF NOT EXISTS idx_memories_agent
            ON memories(agent);
        CREATE INDEX IF NOT EXISTS idx_memories_created
            ON memories(created_at);

        CREATE TABLE IF NOT EXISTS agents (
            name                TEXT PRIMARY KEY,
            memory_count        INTEGER NOT NULL DEFAULT 0,
            last_consolidation  TEXT,
            created_at          TEXT NOT NULL
        );
    """)
    conn.commit()


def store_memory(
    content: str,
    agent: str = "default",
    tags: List[str] = None,
    importance: float = 0.5,
    db_path: Optional[Path] = None,
) -> dict:
    """Store a new memory. Returns the stored memory dict."""
    conn = get_connection(db_path)
    now = datetime.now(timezone.utc).isoformat()
    mem_id = str(uuid.uuid4())[:8]  # short ID for readability
    tags_str = ",".join(tags) if tags else ""

    conn.execute(
        """INSERT INTO memories
           (id, agent, content, tags, importance, created_at, last_accessed, access_count, decay_score)
           VALUES (?, ?, ?, ?, ?, ?, ?, 0, 1.0)""",
        (mem_id, agent, content, tags_str, importance, now, now),
    )

    # Upsert agent stats
    conn.execute(
        """INSERT INTO agents (name, memory_count, created_at)
           VALUES (?, 1, ?)
           ON CONFLICT(name) DO UPDATE SET memory_count = memory_count + 1""",
        (agent, now),
    )
    conn.commit()
    conn.close()

    return {
        "id": mem_id,
        "agent": agent,
        "content": content,
        "tags": tags or [],
        "importance": importance,
        "created_at": now,
        "decay_score": 1.0,
    }


def recall_memories(
    query: str,
    agent: Optional[str] = None,
    limit: int = 5,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """
    Recall memories by text search.
    Step 2: basic FTS via LIKE. Step 3 will add vector similarity.
    """
    conn = get_connection(db_path)
    now = datetime.now(timezone.utc).isoformat()

    # Build query — basic keyword match across content
    words = query.lower().split()
    conditions = " AND ".join(["LOWER(content) LIKE ?" for _ in words])
    params = [f"%{w}%" for w in words]

    if agent:
        conditions = f"agent = ? AND ({conditions})"
        params = [agent] + params

    params.append(limit)
    sql = f"""
        SELECT * FROM memories
        WHERE {conditions}
        ORDER BY decay_score DESC, last_accessed DESC
        LIMIT ?
    """
    rows = conn.execute(sql, params).fetchall()

    # Bump access_count + last_accessed for retrieved memories
    ids = [r["id"] for r in rows]
    if ids:
        placeholders = ",".join("?" * len(ids))
        conn.execute(
            f"""UPDATE memories
                SET access_count = access_count + 1,
                    last_accessed = ?,
                    decay_score = MIN(1.0, decay_score + 0.1)
                WHERE id IN ({placeholders})""",
            [now] + ids,
        )
        conn.commit()

    results = [
        {
            "id": r["id"],
            "agent": r["agent"],
            "content": r["content"],
            "tags": r["tags"].split(",") if r["tags"] else [],
            "importance": r["importance"],
            "decay_score": r["decay_score"],
            "access_count": r["access_count"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]
    conn.close()
    return results


def list_memories(
    agent: Optional[str] = None,
    since_days: Optional[int] = None,
    limit: int = 20,
    db_path: Optional[Path] = None,
) -> List[dict]:
    """List memories, optionally filtered by agent and recency."""
    conn = get_connection(db_path)
    conditions = []
    params = []

    if agent:
        conditions.append("agent = ?")
        params.append(agent)

    if since_days:
        from datetime import timedelta
        cutoff = (datetime.now(timezone.utc) - timedelta(days=since_days)).isoformat()
        conditions.append("created_at >= ?")
        params.append(cutoff)

    where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
    params.append(limit)

    sql = f"SELECT * FROM memories {where} ORDER BY created_at DESC LIMIT ?"
    rows = conn.execute(sql, params).fetchall()
    conn.close()

    return [
        {
            "id": r["id"],
            "agent": r["agent"],
            "content": r["content"],
            "tags": r["tags"].split(",") if r["tags"] else [],
            "importance": r["importance"],
            "decay_score": r["decay_score"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


def forget_memory(mem_id: str, db_path: Optional[Path] = None) -> bool:
    """Delete a memory by ID. Returns True if found and deleted."""
    conn = get_connection(db_path)
    cursor = conn.execute("DELETE FROM memories WHERE id = ?", (mem_id,))
    conn.commit()
    conn.close()
    return cursor.rowcount > 0


def get_stats(db_path: Optional[Path] = None) -> dict:
    """Return basic stats about the memory DB."""
    conn = get_connection(db_path)
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    agents_count = conn.execute("SELECT COUNT(*) FROM agents").fetchone()[0]
    agents = conn.execute(
        "SELECT agent, COUNT(*) as cnt FROM memories GROUP BY agent"
    ).fetchall()
    db = db_path or DB_PATH
    size_kb = round(Path(db).stat().st_size / 1024, 1) if Path(db).exists() else 0
    conn.close()
    return {
        "total_memories": total,
        "agents_count": agents_count,
        "agents": {r["agent"]: r["cnt"] for r in agents},
        "db_size_kb": size_kb,
    }
