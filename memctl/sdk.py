"""Python SDK for memctl — `from memctl import Memory`."""

from pathlib import Path
from typing import List, Optional

from memctl.db import (
    store_memory,
    recall_memories,
    list_memories,
    forget_memory,
    get_stats,
)
from memctl.decay import run_decay
from memctl.consolidation import consolidate


class Memory:
    """High-level Python SDK for memctl.

    Example:
        from memctl import Memory

        mem = Memory(agent="myagent")
        mem.store("Polar API expects price in cents", tags=["api", "billing"])
        results = mem.recall("polar pricing", limit=5)
        mem.consolidate()
    """

    def __init__(
        self,
        agent: str = "default",
        db_path: Optional[Path] = None,
    ):
        self.agent = agent
        self.db_path = db_path

    def store(
        self,
        content: str,
        tags: List[str] = None,
        importance: float = 0.5,
    ) -> dict:
        """Store a new memory."""
        return store_memory(
            content,
            agent=self.agent,
            tags=tags or [],
            importance=importance,
            db_path=self.db_path,
        )

    def recall(
        self,
        query: str,
        limit: int = 5,
        cross_agent: bool = False,
    ) -> List[dict]:
        """Recall memories by semantic similarity.

        Args:
            query: Natural language search query
            limit: Max results
            cross_agent: If True, search across all agents (not just self.agent)
        """
        agent = None if cross_agent else self.agent
        return recall_memories(query, agent=agent, limit=limit, db_path=self.db_path)

    def list(
        self,
        since_days: Optional[int] = None,
        limit: int = 20,
    ) -> List[dict]:
        """List memories for this agent."""
        return list_memories(
            agent=self.agent,
            since_days=since_days,
            limit=limit,
            db_path=self.db_path,
        )

    def forget(self, memory_id: str) -> bool:
        """Delete a memory by ID."""
        return forget_memory(memory_id, db_path=self.db_path)

    def stats(self) -> dict:
        """Get memory DB stats."""
        return get_stats(db_path=self.db_path)

    def decay(self, apply: bool = False) -> tuple:
        """Run memory decay.

        Args:
            apply: If True, actually delete low-score memories. Default: dry-run.

        Returns:
            (to_delete, to_update) lists
        """
        return run_decay(dry_run=not apply, db_path=self.db_path)

    def consolidate(
        self,
        threshold: float = 0.85,
        apply: bool = True,
    ) -> List[dict]:
        """Merge redundant memories using LLM consolidation.

        Args:
            threshold: Cosine similarity threshold (default 0.85)
            apply: If True (default), actually merge. False = dry-run.
        """
        return consolidate(
            agent=self.agent,
            dry_run=not apply,
            threshold=threshold,
            db_path=self.db_path,
        )
