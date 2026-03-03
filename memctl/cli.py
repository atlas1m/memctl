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
    table.add_column("Score", width=7, justify="right")

    for r in results:
        table.add_row(
            r["id"],
            r["content"],
            r["agent"],
            ", ".join(r["tags"]) or "—",
            f"{r['decay_score']:.2f}",
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
def stats():
    """Show memory database statistics.

    \b
    Example:
        memctl stats
    """
    s = get_stats()
    console.print(f"\n[bold]memctl stats[/bold]")
    console.print(f"  Total memories : [cyan]{s['total_memories']}[/cyan]")
    console.print(f"  Agents         : [cyan]{s['agents_count']}[/cyan]")
    console.print(f"  DB size        : [cyan]{s['db_size_kb']} KB[/cyan]")

    if s["agents"]:
        console.print("\n  [bold]Memories per agent:[/bold]")
        for agent, count in s["agents"].items():
            console.print(f"    {agent}: {count}")
    console.print()
