"""MCP server for memctl — exposes store, recall, forget, consolidate as MCP tools."""

from mcp.server.fastmcp import FastMCP

from memctl.db import store_memory, recall_memories, forget_memory, get_stats
from memctl.consolidation import consolidate

mcp = FastMCP("memctl")


@mcp.tool()
def store(
    content: str,
    agent: str = "default",
    tags: str = "",
    importance: float = 0.5,
) -> dict:
    """Store a new memory.

    Args:
        content: The fact or information to remember
        agent: Agent namespace (e.g. 'atlas', 'claude-code')
        tags: Comma-separated tags (e.g. 'api,billing')
        importance: Importance score 0.0-1.0 (default 0.5)

    Returns:
        The stored memory with its ID
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    mem = store_memory(content, agent=agent, tags=tag_list, importance=importance)
    return {
        "id": mem["id"],
        "agent": mem["agent"],
        "content": mem["content"],
        "tags": mem["tags"],
        "importance": mem["importance"],
        "created_at": mem["created_at"],
    }


@mcp.tool()
def recall(
    query: str,
    agent: str = None,
    limit: int = 5,
) -> list:
    """Recall memories by semantic similarity.

    Args:
        query: Natural language query to search for
        agent: Filter by agent namespace (optional)
        limit: Maximum number of results (default 5)

    Returns:
        List of matching memories with similarity scores
    """
    results = recall_memories(query, agent=agent, limit=limit)
    return [
        {
            "id": r["id"],
            "content": r["content"],
            "agent": r["agent"],
            "tags": r["tags"],
            "similarity": r.get("similarity", 0),
            "decay_score": r["decay_score"],
        }
        for r in results
    ]


@mcp.tool()
def forget(memory_id: str) -> dict:
    """Delete a memory by ID.

    Args:
        memory_id: The memory ID to delete

    Returns:
        Success status
    """
    deleted = forget_memory(memory_id)
    return {"deleted": deleted, "id": memory_id}


@mcp.tool()
def stats() -> dict:
    """Get memory database statistics.

    Returns:
        Stats including total memories, agents, DB size, vector search status
    """
    return get_stats()


@mcp.tool()
def consolidate_memories(
    agent: str = None,
    threshold: float = 0.85,
) -> dict:
    """Find and merge redundant memories using LLM consolidation.

    Requires ANTHROPIC_API_KEY environment variable for LLM merging.
    Falls back to simple concatenation if key not set.

    Args:
        agent: Agent namespace to consolidate (optional, all agents if not set)
        threshold: Cosine similarity threshold for grouping (default 0.85)

    Returns:
        Summary of consolidation actions
    """
    actions = consolidate(agent=agent, dry_run=False, threshold=threshold)
    return {
        "clusters_merged": len(actions),
        "actions": [
            {
                "agent": a["agent"],
                "original_count": a["original_count"],
                "merged": a["merged"],
            }
            for a in actions
        ],
    }


def serve():
    """Start the MCP server (stdio transport for Claude Desktop / Cursor)."""
    mcp.run(transport="stdio")
