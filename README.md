# memctl

**Persistent memory for AI agents. One SQLite file. Zero infra.**

[![PyPI](https://img.shields.io/pypi/v/memoctl.svg)](https://pypi.org/project/memoctl/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)

> ⚠️ **Status**: Under active development. CLI skeleton coming shortly.

---

## Install

```bash
pip install memoctl
```

That's it. No Docker. No Postgres. No vector database server.

> Note: CLI command is `memctl`. PyPI package is `memoctl` due to a naming conflict with an unrelated package.

---

## Quick start

```bash
# Store a fact
memctl store "Polar API expects price in cents, not euros" --agent myagent --tags api,billing

# Recall by semantic similarity
memctl recall "Polar pricing" --agent myagent

# List recent memories
memctl list --agent myagent --since 7d

# Run memory decay (forget unused memories)
memctl decay --dry-run
memctl decay --apply

# Check stats
memctl stats
```

---

## Why memctl

Your AI agent forgets everything between sessions. The existing solutions are:

- **Mem0** — Needs Docker + Postgres + Qdrant. $19 → $249 pricing gap.
- **LangMem** — Locked to LangChain ecosystem.
- **txt files** — No semantic search. No decay. Memories pile up forever.

memctl is different: **one SQLite file** (`~/.memctl/memory.db`), embedded vector search, and configurable memory decay — so your agent remembers what matters and forgets what doesn't.

---

## Features

- 📦 **Zero infra** — single SQLite file, no services to run
- 🔍 **Semantic recall** — embedded vector search, no external API needed
- 🧠 **Memory decay** — configurable forgetting curves (exponential, linear, step)
- 🔄 **Auto-consolidation** — merges redundant memories via LLM
- 🤖 **Multi-agent** — isolated namespaces per agent, cross-agent recall
- 🔌 **MCP server** — `memctl mcp serve` for Claude Desktop, Cursor, Windsurf
- 🐍 **Python SDK** — `from memctl import Memory`
- 📤 **Export/Import** — JSON format, portable between machines

---

## Memory decay — the killer feature

Memories you don't access fade over time. Configurable in `~/.memctl/config.toml`:

```toml
[decay]
enabled = true
curve = "exponential"   # exponential | linear | step
half_life_days = 30     # 50% score loss after 30 days without access
min_score = 0.1         # below this = deletion candidate

[consolidation]
auto = true
threshold = 0.85        # cosine similarity > 0.85 = merge candidates
```

```bash
$ memctl decay --dry-run
# Would remove 12 memories (score < 0.1):
#   "Vercel deploy command syntax" (last accessed: 47 days ago, score: 0.08)
#   "Old API endpoint format" (last accessed: 61 days ago, score: 0.04)
#   ...

$ memctl decay --apply
✓ Removed 12 memories. DB size: 4.2 MB → 3.8 MB
```

---

## Pricing

| Tier | Price | What you get |
|---|---|---|
| **OSS Core** | Free (MIT) | store, recall, list, forget, export/import, MCP server, embedded vectors |
| **Pro** | $29 one-time | Decay curves, LLM consolidation, `memctl health`, priority support |
| **Cloud Sync** | $9/mo | Cross-machine sync, auto-backup, minimal web dashboard |
| **Team** | $19/mo | Shared memory across agents, RBAC, audit log |

The free tier is **genuinely useful** — not a teaser. Paying unlocks the power features.

---

## Comparison

| | memctl | Mem0 | Zep | LangMem | txt files |
|---|---|---|---|---|---|
| Install | `pip install memoctl` | Docker + Postgres + Qdrant | Docker | pip (LangChain only) | — |
| Infra | Zero | Heavy | Moderate | None | None |
| Semantic search | ✅ | ✅ | ✅ | ✅ | ❌ |
| Memory decay | ✅ | ❌ | ❌ | ❌ | ❌ |
| Auto-consolidation | ✅ | ❌ | ❌ | ❌ | ❌ |
| MCP server | ✅ | ❌ | ❌ | ❌ | ❌ |
| Multi-framework | ✅ | ✅ | ✅ | ❌ | ✅ |
| Price | Free + $29 Pro | $0 → $249/mo | $0 → $$$ | Free | Free |

---

## Architecture

```
memctl CLI
    │
    ▼
~/.memctl/memory.db  (SQLite single file)
    │
    ├── memories table (content, agent, tags, decay_score, embeddings)
    ├── agents table (stats, last_consolidation)
    └── config (decay params, embedding model)
```

Embeddings: local model via `sentence-transformers` by default. Optional: OpenAI/Anthropic embeddings via config.

---

## MCP server

```bash
memctl mcp serve
# Exposes: store, recall, forget, consolidate as MCP tools
# Works with Claude Desktop, Cursor, Windsurf, any MCP-compatible client
```

---

## Python SDK

```python
from memctl import Memory

mem = Memory(agent="myagent")
mem.store("Polar API expects price in cents", tags=["api", "billing"])

results = mem.recall("pricing issue", limit=5)
for r in results:
    print(r.content, r.similarity)

mem.consolidate()
```

---

## Contributing

Issues and PRs welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

This project follows the [MIT License](LICENSE).
