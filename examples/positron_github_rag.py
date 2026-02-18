# /// script
# dependencies = ["PyGithub", "raghilda", "chatlas", "typer", "rich"]
#
# [tool.uv.sources]
# raghilda = { path = "..", editable = true }
# ///
"""
Example: Building a RAG store from GitHub Issues and PRs

Usage:
    uv run examples/positron_github_rag.py sync posit-dev/positron   # Download & ingest
    uv run examples/positron_github_rag.py chat posit-dev/positron   # Interactive chat
    uv run examples/positron_github_rag.py query posit-dev/positron "search term"
    uv run examples/positron_github_rag.py status posit-dev/positron # Show status
"""

import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

import typer
from github import Auth, Github
from rich.console import Console

app = typer.Typer(help="GitHub RAG CLI")
console = Console()


def get_default_db_path(repo: str) -> Path:
    """Get default database path for a repo."""
    return Path(repo.replace("/", "_") + ".db")


def get_default_jsonl_path(repo: str) -> Path:
    """Get default JSONL path for a repo."""
    return Path(repo.replace("/", "_") + ".jsonl")


def get_github_token() -> str:
    """Get GitHub token from environment or gh CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError("No GitHub token found. Set GITHUB_TOKEN or run 'gh auth login'")


def get_metadata_path(db_path: Path) -> Path:
    """Get the metadata file path for a database."""
    return db_path.with_suffix(".meta.json")


def load_metadata(db_path: Path) -> dict:
    """Load sync metadata."""
    metadata_file = get_metadata_path(db_path)
    if metadata_file.exists():
        return json.loads(metadata_file.read_text())
    return {}


def save_metadata(db_path: Path, metadata: dict):
    """Save sync metadata."""
    metadata_file = get_metadata_path(db_path)
    metadata_file.write_text(json.dumps(metadata, indent=2, default=str))


def load_issues_from_jsonl(jsonl_path: Path) -> dict[int, dict]:
    """Load issues from JSONL file, keyed by issue number."""
    issues = {}
    if jsonl_path.exists():
        with open(jsonl_path) as f:
            for line in f:
                if line.strip():
                    issue = json.loads(line)
                    issues[issue["number"]] = issue
    return issues


def save_issues_to_jsonl(jsonl_path: Path, issues: dict[int, dict]):
    """Save issues to JSONL file."""
    with open(jsonl_path, "w") as f:
        for issue in issues.values():
            f.write(json.dumps(issue) + "\n")


def issue_to_dict(issue) -> dict:
    """Convert a PyGithub issue to a serializable dict."""
    comments = []
    for comment in issue.get_comments():
        comments.append({
            "author": comment.user.login if comment.user else None,
            "body": comment.body,
            "created_at": comment.created_at.isoformat(),
        })
    return {
        "number": issue.number,
        "title": issue.title,
        "body": issue.body,
        "url": issue.html_url,
        "state": issue.state,
        "author": issue.user.login if issue.user else None,
        "labels": [label.name for label in issue.labels],
        "created_at": issue.created_at.isoformat(),
        "updated_at": issue.updated_at.isoformat(),
        "is_pull_request": issue.pull_request is not None,
        "comments": comments,
    }


def prepare_issue(issue: dict):
    """Convert an issue dict to a Document with chunks."""
    from raghilda.document import MarkdownDocument
    from raghilda.chunker import MarkdownChunker

    # Use smaller chunks to handle code-heavy issues (code tokenizes inefficiently)
    chunker = MarkdownChunker(chunk_size=800)
    item_type = "PR" if issue.get("is_pull_request") else "Issue"
    labels = ", ".join(issue.get("labels", []))

    lines = [
        f"# {item_type} #{issue['number']}: {issue['title']}",
        "",
        f"**State:** {issue['state']}",
        f"**Author:** {issue.get('author') or 'unknown'}",
        f"**Labels:** {labels or 'none'}",
        "",
    ]

    body = issue.get("body") or ""
    if body.strip():
        lines.append(body)
        lines.append("")

    for comment in issue.get("comments", []):
        comment_body = comment.get("body") or ""
        if comment_body.strip():
            comment_author = comment.get("author") or "unknown"
            lines.append(f"## Comment by {comment_author}")
            lines.append("")
            lines.append(comment_body)
            lines.append("")

    content = "\n".join(lines)
    doc = MarkdownDocument(content=content, origin=issue["url"])
    doc = chunker.chunk_document(doc)
    return doc


@app.command()
def sync(
    repo: Annotated[str, typer.Argument(help="GitHub repo (e.g., posit-dev/positron)")],
    db_path: Annotated[Path | None, typer.Option(help="Database path")] = None,
    jsonl_path: Annotated[Path | None, typer.Option(help="JSONL cache path")] = None,
):
    """Download and ingest issues from GitHub (incremental).

    Issues are cached in a JSONL file. Only new/updated issues are fetched
    from GitHub, then the store is rebuilt from the complete cache.
    """
    from raghilda.store import DuckDBStore
    from raghilda.embedding import EmbeddingOpenAI

    if db_path is None:
        db_path = get_default_db_path(repo)
    if jsonl_path is None:
        jsonl_path = get_default_jsonl_path(repo)

    # Load existing issues from JSONL cache
    issues = load_issues_from_jsonl(jsonl_path)
    console.print(f"[cyan]Loaded {len(issues)} issues from cache[/cyan]")

    # Fetch new/updated issues from GitHub
    token = get_github_token()
    g = Github(auth=Auth.Token(token))
    github_repo = g.get_repo(repo)

    metadata = load_metadata(db_path)
    last_update = metadata.get("last_update")

    if last_update:
        console.print(f"[cyan]Fetching updates since {last_update}...[/cyan]")
        since = datetime.fromisoformat(last_update)
        issues_iter = github_repo.get_issues(state="all", sort="updated", since=since)
    else:
        console.print("[cyan]Fetching all issues...[/cyan]")
        issues_iter = github_repo.get_issues(state="all", sort="updated")

    # Merge new issues into cache
    update_time = datetime.now(timezone.utc)
    new_count = 0
    for issue in issues_iter:
        issue_dict = issue_to_dict(issue)
        issues[issue_dict["number"]] = issue_dict
        new_count += 1
        if new_count % 100 == 0:
            console.print(f"[dim]Fetched {new_count} issues...[/dim]")

    console.print(f"[cyan]Fetched {new_count} new/updated issues[/cyan]")

    # Save updated cache
    save_issues_to_jsonl(jsonl_path, issues)
    console.print(f"[cyan]Saved {len(issues)} issues to cache[/cyan]")

    # Rebuild store from cache
    console.print(f"[cyan]Building store at {db_path}...[/cyan]")
    store = DuckDBStore.create(
        location=str(db_path),
        embed=EmbeddingOpenAI(),
        overwrite=True,
        name="github_issues",
        title=f"GitHub Issues: {repo}",
    )

    store.ingest(issues.values(), prepare=prepare_issue)

    console.print("[cyan]Building search indexes...[/cyan]")
    store.build_index()

    # Save metadata
    metadata["last_update"] = update_time.isoformat()
    save_metadata(db_path, metadata)

    console.print(f"[green]Done! Store contains {store.size()} documents.[/green]")


@app.command()
def chat(
    repo: Annotated[str, typer.Argument(help="GitHub repo (e.g., posit-dev/positron)")],
    db_path: Annotated[Path | None, typer.Option(help="Database path")] = None,
):
    """Interactive chat with RAG context."""
    from raghilda.store import DuckDBStore
    from chatlas import ChatOpenAI

    if db_path is None:
        db_path = get_default_db_path(repo)

    if not db_path.exists():
        console.print(f"[red]No database found at {db_path}. Run 'sync' first.[/red]")
        raise typer.Exit(1)

    store = DuckDBStore.connect(location=str(db_path))

    def retrieve(query: str, top_k: int = 5) -> str:
        """Search the Positron GitHub issues and PRs for relevant information."""
        chunks = store.retrieve(query, top_k=top_k)
        results = []
        for chunk in chunks:
            results.append({"text": chunk.text, "context": chunk.context})
        return json.dumps(results)

    chat_session = ChatOpenAI()
    chat_session.register_tool(retrieve)
    chat_session.console()


@app.command()
def query(
    repo: Annotated[str, typer.Argument(help="GitHub repo (e.g., posit-dev/positron)")],
    search_query: Annotated[str, typer.Argument(help="Query to search for")],
    db_path: Annotated[Path | None, typer.Option(help="Database path")] = None,
    top_k: Annotated[int, typer.Option(help="Number of results")] = 5,
):
    """One-off query against the RAG store."""
    from raghilda.store import DuckDBStore

    if db_path is None:
        db_path = get_default_db_path(repo)

    if not db_path.exists():
        console.print(f"[red]No database found at {db_path}. Run 'sync' first.[/red]")
        raise typer.Exit(1)

    store = DuckDBStore.connect(location=str(db_path))
    chunks = store.retrieve(search_query, top_k=top_k)

    if not chunks:
        console.print("[yellow]No results found.[/yellow]")
        return

    for i, chunk in enumerate(chunks, 1):
        console.print(f"\n[bold cyan]Result {i}[/bold cyan]")
        console.print(f"[dim]{chunk.context}[/dim]")
        console.print(chunk.text[:500] + "..." if len(chunk.text) > 500 else chunk.text)


@app.command()
def status(
    repo: Annotated[str, typer.Argument(help="GitHub repo (e.g., posit-dev/positron)")],
    db_path: Annotated[Path | None, typer.Option(help="Database path")] = None,
):
    """Show current database status."""
    from rich.table import Table

    if db_path is None:
        db_path = get_default_db_path(repo)

    table = Table(title=f"GitHub RAG Status: {repo}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    metadata = load_metadata(db_path)

    table.add_row("Database path", str(db_path.absolute()))
    table.add_row("Database exists", "Yes" if db_path.exists() else "No")

    if metadata:
        table.add_row("Last sync", metadata.get("last_update", "N/A"))

    if db_path.exists():
        from raghilda.store import DuckDBStore
        store = DuckDBStore.connect(location=str(db_path))
        table.add_row("Documents in store", str(store.size()))
        db_size = db_path.stat().st_size / (1024 * 1024)
        table.add_row("Database size", f"{db_size:.2f} MB")

    console.print(table)


if __name__ == "__main__":
    app()
