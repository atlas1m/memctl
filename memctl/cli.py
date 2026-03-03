"""memctl CLI — Persistent memory for AI agents."""

import click
from rich.console import Console
from rich.table import Table
from rich import print as rprint

from memctl import __version__
from memctl.db import (
    store_memory,
    recall_memories,
    list_memories,
    forget_memory,
    get_stats,
)
from memctl.decay import run_decay, write_default_config
from memctl.consolidation import consolidate as run_consolidate

console = Console()


@click.group()
@click.version_option(__version__, prog_name="memctl")
def main():
    """memctl — Persistent memory for AI agents.

    One SQLite file. Zero infra. pip install memoctl
    """


@main.command()
@click.argument("content")
@click.option("--agent", "-a", default="default", help="Agent namespace")
@click.option("--tags", "-t", default="", help="Comma-separated tags")
@click.option("--importance", "-i", default=0.5, type=float, help="Importance 0-1")
def store(content: str, agent: str, tags: str, importance: float):
    """Store a new memory.

    \b
    Example:
        memctl store "Polar API expects price in cents" --agent atlas --tags api,billing
    """
    tag_list = [t.strip() for t in tags.split(",") if t.strip()]
    mem = store_memory(content, agent=agent, tags=tag_list, importance=importance)
    console.print(
        f"[green]✓[/green] Stored [{mem['id']}] "
        f"(agent: {mem['agent']}, importance: {mem['importance']:.1f})"
    )


@main.command()
@click.argument("query")
@click.option("--agent", "-a", default=None, help="Filter by agent namespace")
@click.option("--limit", "-n", default=5, help="Number of results")
def recall(query: str, agent: str, limit: int):
    """Recall memories by semantic similarity.

    \b
    Example:
        memctl recall "polar pricing" --agent atlas --limit 5
    """
    results = recall_memories(query, agent=agent, limit=limit)
    if not results:
        console.print("[yellow]No memories found matching your query.[/yellow]")
        return

    table = Table(title=f"Memories matching: '{query}'", show_header=True)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Content")
    table.add_column("Agent", width=12)
    table.add_column("Tags", width=16)
    table.add_column("Sim", width=6, justify="right")

    for r in results:
        table.add_row(
            r["id"],
            r["content"],
            r["agent"],
            ", ".join(r["tags"]) or "—",
            f"{r.get('similarity', r['decay_score']):.2f}",
        )

    console.print(table)


@main.command(name="list")
@click.option("--agent", "-a", default=None, help="Filter by agent namespace")
@click.option("--since", "-s", default=None, type=int, help="Since N days ago")
@click.option("--limit", "-n", default=20, help="Max results")
def list_cmd(agent: str, since: int, limit: int):
    """List stored memories.

    \b
    Example:
        memctl list --agent atlas --since 7
    """
    results = list_memories(agent=agent, since_days=since, limit=limit)
    if not results:
        console.print("[yellow]No memories found.[/yellow]")
        return

    table = Table(title="Memories", show_header=True)
    table.add_column("ID", style="dim", width=10)
    table.add_column("Agent", width=12)
    table.add_column("Content")
    table.add_column("Tags", width=16)
    table.add_column("Created", width=20)

    for r in results:
        table.add_row(
            r["id"],
            r["agent"],
            r["content"][:80] + ("…" if len(r["content"]) > 80 else ""),
            ", ".join(r["tags"]) or "—",
            r["created_at"][:16].replace("T", " "),
        )

    console.print(table)


@main.command()
@click.argument("memory_id")
def forget(memory_id: str):
    """Delete a memory by ID.

    \b
    Example:
        memctl forget abc12345
    """
    if forget_memory(memory_id):
        console.print(f"[green]✓[/green] Memory [{memory_id}] deleted.")
    else:
        console.print(f"[red]✗[/red] Memory [{memory_id}] not found.")


@main.command()
@click.option("--dry-run", is_flag=True, default=True, help="Preview only (default)")
@click.option("--apply", "apply_decay", is_flag=True, default=False, help="Actually apply decay")
def decay(dry_run: bool, apply_decay: bool):
    """Run memory decay — forget unused memories.

    \b
    Examples:
        memctl decay --dry-run   # preview what would be forgotten
        memctl decay --apply     # apply decay (remove low-score memories)
    """
    # --apply overrides --dry-run
    is_dry = not apply_decay

    to_delete, to_update = run_decay(dry_run=is_dry)

    if not to_delete and not to_update:
        console.print("[green]✓[/green] No decay needed — all memories are fresh.")
        return

    mode = "[dim](dry run)[/dim]" if is_dry else "[bold red](applied)[/bold red]"

    if to_delete:
        table = Table(title=f"Memories to remove {mode}", show_header=True)
        table.add_column("ID", style="dim", width=10)
        table.add_column("Agent", width=12)
        table.add_column("Content")
        table.add_column("Score", width=12, justify="right")
        table.add_column("Last accessed", width=18)

        for entry in to_delete:
            table.add_row(
                entry["id"],
                entry["agent"],
                entry["content"],
                f"{entry['old_score']:.2f} → [red]{entry['new_score']:.2f}[/red]",
                entry["last_accessed"],
            )
        console.print(table)
        action = "Would remove" if is_dry else "Removed"
        console.print(f"\n{action} [red]{len(to_delete)}[/red] memories.")

    if to_update:
        console.print(f"\n[dim]Score updated for {len(to_update)} memories (still above threshold).[/dim]")

    if is_dry:
        console.print("\n[dim]Run with --apply to actually apply decay.[/dim]")

    # Write default config if not present
    write_default_config()


@main.command()
@click.option("--agent", "-a", default=None, help="Agent namespace to consolidate")
@click.option("--dry-run", is_flag=True, default=True, help="Preview only (default)")
@click.option("--apply", "apply_consolidate", is_flag=True, default=False, help="Actually merge")
@click.option("--threshold", "-t", default=None, type=float, help="Similarity threshold (default: 0.85)")
def consolidate(agent: str, dry_run: bool, apply_consolidate: bool, threshold: float):
    """Merge redundant memories using LLM (Claude Haiku).

    Requires ANTHROPIC_API_KEY for LLM merging.
    Falls back to simple concatenation if key not set.

    \b
    Examples:
        memctl consolidate --agent atlas --dry-run
        memctl consolidate --agent atlas --apply
    """
    is_dry = not apply_consolidate
    with console.status("[bold]Scanning for similar memories...[/bold]"):
        actions = run_consolidate(
            agent=agent,
            dry_run=is_dry,
            threshold=threshold,
        )

    if not actions:
        console.print("[green]✓[/green] No redundant memories found.")
        return

    mode = "[dim](dry run)[/dim]" if is_dry else "[bold green](applied)[/bold green]"
    console.print(f"\n[bold]Found {len(actions)} consolidation(s) {mode}[/bold]\n")

    for i, action in enumerate(actions, 1):
        console.print(f"[bold]Cluster {i}[/bold] — {action['agent']} ({action['original_count']} memories → 1):")
        for orig in action["original_texts"]:
            console.print(f"  [red]- {orig[:100]}[/red]")
        console.print(f"  [green]→ {action['merged']}[/green]\n")

    if is_dry:
        console.print("[dim]Run with --apply to merge. Requires ANTHROPIC_API_KEY for LLM consolidation.[/dim]")


@main.command()
def agents():
    """List all agents and their memory stats.

    \b
    Example:
        memctl agents
    """
    s = get_stats()
    if not s["agents"]:
        console.print("[yellow]No agents found. Use --agent <name> to create one.[/yellow]")
        return

    table = Table(title="Registered Agents", show_header=True)
    table.add_column("Agent", style="bold")
    table.add_column("Memories", justify="right", width=10)

    for agent_name, count in sorted(s["agents"].items(), key=lambda x: -x[1]):
        table.add_row(agent_name, str(count))

    console.print(table)
    console.print(f"\n  Total memories: [cyan]{s['total_memories']}[/cyan] across {s['agents_count']} agent(s)")


@main.command()
def stats():
    """Show memory database statistics.

    \b
    Example:
        memctl stats
    """
    s = get_stats()
    console.print(f"\n[bold]memctl stats[/bold]")
    console.print(f"  Total memories : [cyan]{s['total_memories']}[/cyan]")
    console.print(f"  With embeddings: [cyan]{s.get('with_embeddings', 0)}[/cyan]")
    console.print(f"  Vector search  : [cyan]{'✓' if s.get('vector_search') else '✗ (fastembed needed)'}[/cyan]")
    console.print(f"  Agents         : [cyan]{s['agents_count']}[/cyan]")
    console.print(f"  DB size        : [cyan]{s['db_size_kb']} KB[/cyan]")

    if s["agents"]:
        console.print("\n  [bold]Memories per agent:[/bold]")
        for agent, count in s["agents"].items():
            console.print(f"    {agent}: {count}")
    console.print()


@main.group()
def mcp():
    """MCP server commands (for Claude Desktop, Cursor, Windsurf)."""


@mcp.command("serve")
def mcp_serve():
    """Start the MCP server (stdio transport).

    \b
    Add to claude_desktop_config.json:
        {
          "mcpServers": {
            "memctl": {
              "command": "memctl",
              "args": ["mcp", "serve"]
            }
          }
        }
    """
    from memctl.mcp_server import serve
    serve()
