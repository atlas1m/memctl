"""Unit tests for memctl CRUD operations (memory.db layer)."""

import tempfile
from pathlib import Path

import pytest

from memctl.db import (
    forget_memory,
    get_connection,
    get_stats,
    list_memories,
    recall_memories,
    store_memory,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Return a temporary DB path isolated per test."""
    return tmp_path / "test_memory.db"


# ---------------------------------------------------------------------------
# store_memory
# ---------------------------------------------------------------------------

class TestStoreMemory:
    def test_returns_dict_with_required_keys(self, tmp_db):
        result = store_memory("A test memory", db_path=tmp_db)
        assert "id" in result
        assert "agent" in result
        assert "content" in result
        assert "created_at" in result
        assert "decay_score" in result

    def test_content_persisted(self, tmp_db):
        store_memory("Persistent fact", db_path=tmp_db)
        memories = list_memories(db_path=tmp_db)
        assert any("Persistent fact" in m["content"] for m in memories)

    def test_default_agent_is_default(self, tmp_db):
        result = store_memory("Hello", db_path=tmp_db)
        assert result["agent"] == "default"

    def test_custom_agent(self, tmp_db):
        result = store_memory("Agent fact", agent="atlas", db_path=tmp_db)
        assert result["agent"] == "atlas"

    def test_tags_stored(self, tmp_db):
        result = store_memory("Tagged memory", tags=["api", "billing"], db_path=tmp_db)
        assert "api" in result["tags"]
        assert "billing" in result["tags"]

    def test_importance_stored(self, tmp_db):
        result = store_memory("Important fact", importance=0.9, db_path=tmp_db)
        assert abs(result["importance"] - 0.9) < 0.01

    def test_decay_score_starts_at_1(self, tmp_db):
        result = store_memory("Fresh memory", db_path=tmp_db)
        assert result["decay_score"] == 1.0

    def test_id_is_unique(self, tmp_db):
        r1 = store_memory("Memory one", db_path=tmp_db)
        r2 = store_memory("Memory two", db_path=tmp_db)
        assert r1["id"] != r2["id"]

    def test_multiple_stores(self, tmp_db):
        for i in range(10):
            store_memory(f"Fact {i}", db_path=tmp_db)
        memories = list_memories(db_path=tmp_db)
        assert len(memories) == 10

    def test_empty_tags(self, tmp_db):
        result = store_memory("No tags", tags=[], db_path=tmp_db)
        assert result["tags"] == []


# ---------------------------------------------------------------------------
# list_memories
# ---------------------------------------------------------------------------

class TestListMemories:
    def test_returns_list(self, tmp_db):
        result = list_memories(db_path=tmp_db)
        assert isinstance(result, list)

    def test_empty_db_returns_empty(self, tmp_db):
        result = list_memories(db_path=tmp_db)
        assert result == []

    def test_returns_all_stored(self, tmp_db):
        store_memory("A", db_path=tmp_db)
        store_memory("B", db_path=tmp_db)
        store_memory("C", db_path=tmp_db)
        result = list_memories(db_path=tmp_db)
        assert len(result) == 3

    def test_filter_by_agent(self, tmp_db):
        store_memory("Forge fact", agent="forge", db_path=tmp_db)
        store_memory("Atlas fact", agent="atlas", db_path=tmp_db)
        result = list_memories(agent="forge", db_path=tmp_db)
        assert len(result) == 1
        assert result[0]["agent"] == "forge"

    def test_limit_respected(self, tmp_db):
        for i in range(10):
            store_memory(f"Fact {i}", db_path=tmp_db)
        result = list_memories(limit=3, db_path=tmp_db)
        assert len(result) == 3

    def test_most_recent_first(self, tmp_db):
        store_memory("Old fact", db_path=tmp_db)
        store_memory("New fact", db_path=tmp_db)
        result = list_memories(db_path=tmp_db)
        assert result[0]["content"] == "New fact"


# ---------------------------------------------------------------------------
# recall_memories
# ---------------------------------------------------------------------------

class TestRecallMemories:
    def test_returns_list(self, tmp_db):
        result = recall_memories("anything", db_path=tmp_db)
        assert isinstance(result, list)

    def test_empty_db_returns_empty(self, tmp_db):
        result = recall_memories("polar api", db_path=tmp_db)
        assert result == []

    def test_keyword_match(self, tmp_db):
        store_memory("Polar API expects price in cents", db_path=tmp_db)
        store_memory("Redis is a fast in-memory store", db_path=tmp_db)
        results = recall_memories("polar api", db_path=tmp_db)
        # Should find the polar memory (keyword fallback)
        contents = [r["content"] for r in results]
        assert any("Polar" in c for c in contents)

    def test_agent_filter(self, tmp_db):
        store_memory("Atlas knows strategy", agent="atlas", db_path=tmp_db)
        store_memory("Forge knows code", agent="forge", db_path=tmp_db)
        results = recall_memories("strategy", agent="atlas", db_path=tmp_db)
        agents = {r["agent"] for r in results}
        assert "forge" not in agents

    def test_limit_respected(self, tmp_db):
        for i in range(10):
            store_memory(f"Fact about topic {i}", db_path=tmp_db)
        results = recall_memories("topic", limit=3, db_path=tmp_db)
        assert len(results) <= 3

    def test_similarity_key_present(self, tmp_db):
        store_memory("Some fact", db_path=tmp_db)
        results = recall_memories("fact", db_path=tmp_db)
        if results:
            assert "similarity" in results[0]


# ---------------------------------------------------------------------------
# forget_memory
# ---------------------------------------------------------------------------

class TestForgetMemory:
    def test_returns_true_on_delete(self, tmp_db):
        mem = store_memory("To be forgotten", db_path=tmp_db)
        result = forget_memory(mem["id"], db_path=tmp_db)
        assert result is True

    def test_returns_false_on_missing_id(self, tmp_db):
        result = forget_memory("nonexistent", db_path=tmp_db)
        assert result is False

    def test_memory_gone_after_forget(self, tmp_db):
        mem = store_memory("Temporary fact", db_path=tmp_db)
        forget_memory(mem["id"], db_path=tmp_db)
        memories = list_memories(db_path=tmp_db)
        ids = [m["id"] for m in memories]
        assert mem["id"] not in ids

    def test_only_target_deleted(self, tmp_db):
        m1 = store_memory("Keep this", db_path=tmp_db)
        m2 = store_memory("Delete this", db_path=tmp_db)
        forget_memory(m2["id"], db_path=tmp_db)
        remaining = list_memories(db_path=tmp_db)
        ids = [m["id"] for m in remaining]
        assert m1["id"] in ids
        assert m2["id"] not in ids


# ---------------------------------------------------------------------------
# get_stats
# ---------------------------------------------------------------------------

class TestGetStats:
    def test_empty_db(self, tmp_db):
        stats = get_stats(db_path=tmp_db)
        assert stats["total_memories"] == 0
        assert stats["agents_count"] == 0

    def test_counts_correctly(self, tmp_db):
        store_memory("A", agent="atlas", db_path=tmp_db)
        store_memory("B", agent="atlas", db_path=tmp_db)
        store_memory("C", agent="forge", db_path=tmp_db)
        stats = get_stats(db_path=tmp_db)
        assert stats["total_memories"] == 3
        assert stats["agents_count"] == 2
        assert stats["agents"]["atlas"] == 2
        assert stats["agents"]["forge"] == 1

    def test_db_size_kb_present(self, tmp_db):
        store_memory("Size test", db_path=tmp_db)
        stats = get_stats(db_path=tmp_db)
        assert "db_size_kb" in stats
        assert stats["db_size_kb"] > 0


# ---------------------------------------------------------------------------
# schema / connection
# ---------------------------------------------------------------------------

class TestConnection:
    def test_schema_created(self, tmp_db):
        conn = get_connection(tmp_db)
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {r["name"] for r in tables}
        assert "memories" in table_names
        assert "agents" in table_names
        conn.close()

    def test_idempotent_schema(self, tmp_db):
        """Calling get_connection twice should not error."""
        get_connection(tmp_db).close()
        get_connection(tmp_db).close()
