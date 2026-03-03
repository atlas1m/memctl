# memctl

**Persistent memory for AI agents.** SQLite-backed, vector search, memory decay. Zero infra.

```bash
pip install memoctl[embeddings]
```

---

## The problem

AI coding agents are amnesiac. Every session starts from zero. Claude Code uses a `claude-progress.txt` as makeshift memory. Cursor loses your project context overnight. Mem0 requires Docker + Postgres + Qdrant.

**memctl** is the SQLite of agent memory: one file, works anywhere, no server.

---

## Install

```bash
# Core (keyword search)
pip install memoctl

# With vector search (recommended — ~25MB model download on first use)
pip install memoctl[embeddings]

# Everything (MCP server + vector search)
pip install memoctl[all]
```

---

## Quick start

```bash
# Store a memory
memctl store "Polar API expects price in cents, not euros"

# Recall by semantic similarity
memctl recall "polar payment"
# ┌────────────┬──────────────────────────────────────┬────────┐
# │ ID         │ Content                              │  Sim   │
# ├────────────┼──────────────────────────────────────┼────────┤
# │ c50ad1d2   │ Polar API expects price in cents...  │  0.78  │
# └────────────┴──────────────────────────────────────┴────────┘

# List all memories
memctl list

# Health check
memctl health
```

---

## Python SDK

```python
from memctl import Memory

mem = Memory(agent="my-agent")

# Store
mem.store("Board Advisor has authority over strategy", tags=["rules", "governance"])

# Recall — semantic similarity
results = mem.recall("who decides strategy", limit=5)
for r in results:
    print(f"[{r['similarity']:.2f}] {r['content']}")

# Decay — forget stale memories
to_delete, _ = mem.decay(apply=False)  # dry-run

# Consolidate — merge duplicates via LLM
mem.consolidate(threshold=0.85)
```

---

## MCP server (Claude Desktop, Cursor, Windsurf)

```bash
memctl mcp serve
```

Add to `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "memctl": {
      "command": "memctl",
      "args": ["mcp", "serve"]
    }
  }
}
```

Tools available: `store`, `recall`, `forget`, `stats`, `consolidate_memories`

---

## Multi-agent namespaces

```bash
memctl store "deploy with vercel --prod" --agent my-coder
memctl store "always reply to Board Advisor" --agent atlas

memctl list --agent my-coder
memctl agents  # list all namespaces
```

---

## Memory decay

Memories decay exponentially when not accessed. Configurable in `~/.memctl/config.toml`:

```toml
[decay]
enabled = true
curve = "exponential"   # exponential | linear | step
half_life_days = 30
min_score = 0.1

[consolidation]
threshold = 0.85
auto = false
```

```bash
memctl decay --dry-run   # preview what would be forgotten
memctl decay --apply     # remove low-score memories
```

---

## Commands

| Command | Description |
|---|---|
| `memctl store <text>` | Store a memory |
| `memctl recall <query>` | Semantic recall |
| `memctl list` | List memories |
| `memctl forget <id>` | Delete a memory |
| `memctl agents` | List agent namespaces |
| `memctl decay` | Run memory decay |
| `memctl consolidate` | Merge redundant memories (LLM) |
| `memctl health` | Diagnostic check |
| `memctl stats` | DB statistics |
| `memctl mcp serve` | Start MCP server |
| `memctl license set <key>` | Activate Pro license |

---

## Free vs Pro

| Feature | Free | Pro ($29) |
|---|---|---|
| store / recall / list / forget | ✓ | ✓ |
| Vector search (fastembed) | ✓ | ✓ |
| Multi-agent namespaces | ✓ | ✓ |
| MCP server | ✓ | ✓ |
| Export / Import | ✓ | ✓ |
| Memory decay (default config) | ✓ | ✓ |
| Custom decay curves (TOML config) | — | ✓ |
| LLM consolidation (Claude Haiku) | — | ✓ |
| memctl health diagnostics | — | ✓ |
| Priority support | — | ✓ |

**[→ Get Pro](https://polar.sh/atlas1m)**

---

## How it works

- **Storage**: SQLite single file at `~/.memctl/memory.db`
- **Vectors**: BAAI/bge-small-en-v1.5 (384 dims) via fastembed — runs fully local, no API
- **Decay**: exponential curve `S(t) = S₀ × 2^(-t/T½)` — recalled memories get a score boost
- **Consolidation**: cosine similarity clustering → Claude Haiku merges clusters into single memories
- **MCP**: FastMCP stdio transport, compatible with Claude Desktop, Cursor, Windsurf

---

## License

MIT. Built by [Atlas](https://atlas1m.com).
