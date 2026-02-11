# /// script
# dependencies = ["PyGithub", "raghilda", "chatlas"]
#
# [tool.uv.sources]
# raghilda = { path = "..", editable = true }
# ///
"""
Example: Building a RAG store from Positron GitHub Issues, PRs, and Discussions

Two-phase approach:
1. Download: Fetch GitHub data to JSON files (supports incremental updates)
2. Ingest: Build RAG store from JSON files

Requirements:
- OpenAI API key set in OPENAI_API_KEY environment variable
- GitHub token set in GITHUB_TOKEN environment variable (or use gh auth token)

Usage:
    # Download all issues (or update existing dump)
    uv run examples/positron_github_rag.py download

    # Ingest from JSON into RAG store
    uv run examples/positron_github_rag.py ingest

    # Do both
    uv run examples/positron_github_rag.py download ingest

    # Chat with the RAG store using ellmer
    uv run examples/positron_github_rag.py chat
"""

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from github import Github

# Configuration
REPO = "posit-dev/positron"
DATA_DIR = Path("positron_data")
ISSUES_FILE = DATA_DIR / "issues.jsonl"
METADATA_FILE = DATA_DIR / "metadata.json"
DB_PATH = "positron_github.db"


def get_github_token() -> str:
    """Get GitHub token from environment or gh CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError("No GitHub token found. Set GITHUB_TOKEN or run 'gh auth login'")


def load_metadata() -> dict:
    """Load download metadata (last update time, etc.)."""
    if METADATA_FILE.exists():
        return json.loads(METADATA_FILE.read_text())
    return {}


def save_metadata(metadata: dict):
    """Save download metadata."""
    METADATA_FILE.write_text(json.dumps(metadata, indent=2, default=str))


def load_existing_issues() -> dict[int, dict]:
    """Load existing issues from JSONL file, keyed by issue number."""
    if ISSUES_FILE.exists():
        issues = {}
        with open(ISSUES_FILE) as f:
            for line in f:
                if line.strip():
                    issue = json.loads(line)
                    issues[issue["number"]] = issue
        return issues
    return {}


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


def download_issues():
    """Download issues from GitHub, with incremental update support."""
    DATA_DIR.mkdir(exist_ok=True)

    token = get_github_token()
    g = Github(token)
    repo = g.get_repo(REPO)

    # Load existing data
    existing = load_existing_issues()
    metadata = load_metadata()
    last_update = metadata.get("last_update")

    if last_update and existing:
        print(f"Incremental update since {last_update}")
        since = datetime.fromisoformat(last_update)
        issues_iter = repo.get_issues(state="all", sort="updated", since=since)
    else:
        print("Full download of all issues...")
        issues_iter = repo.get_issues(state="all", sort="updated")

    # Fetch and write issues as we go
    update_time = datetime.now(timezone.utc)
    count = 0

    try:
        for issue in issues_iter:
            existing[issue.number] = issue_to_dict(issue)
            count += 1
            if count % 100 == 0:
                print(f"  Fetched {count} issues...")
                # Periodic save
                _save_issues(existing, metadata, update_time)
    finally:
        # Always save on exit (even if interrupted)
        _save_issues(existing, metadata, update_time)
        print(f"  Fetched {count} new/updated issues")
        print(f"  Total issues: {len(existing)}")
        print(f"Saved to {ISSUES_FILE}")


def _save_issues(existing: dict, metadata: dict, update_time: datetime):
    """Save issues to JSONL and update metadata."""
    all_issues = sorted(existing.values(), key=lambda x: x["number"], reverse=True)
    with open(ISSUES_FILE, "w") as f:
        for issue in all_issues:
            f.write(json.dumps(issue) + "\n")

    metadata["last_update"] = update_time.isoformat()
    metadata["total_issues"] = len(all_issues)
    save_metadata(metadata)


def prepare_issue(issue: dict):
    """Convert an issue dict to a Document with chunks for body + comments."""
    from raghilda.document import MarkdownDocument
    from raghilda.chunker import MarkdownChunker

    chunker = MarkdownChunker()

    item_type = "PR" if issue.get("is_pull_request") else "Issue"
    labels = ", ".join(issue.get("labels", []))

    metadata = {
        "number": issue["number"],
        "state": issue["state"],
        "author": issue.get("author"),
        "is_pull_request": issue.get("is_pull_request", False),
        "labels": issue.get("labels", []),
    }

    # Build content
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
    doc = MarkdownDocument(content=content, origin=issue["url"], metadata=metadata)
    doc = chunker.chunk_document(doc)

    return doc


def ingest_issues():
    """Ingest issues from JSON into RAG store."""
    from raghilda.store import DuckDBStore
    from raghilda.embedding import EmbeddingOpenAI

    if not ISSUES_FILE.exists():
        print(f"No issues file found at {ISSUES_FILE}. Run 'download' first.")
        return

    print(f"Creating store at {DB_PATH}...")
    store = DuckDBStore.create(
        location=DB_PATH,
        embed=EmbeddingOpenAI(),
        overwrite=True,
        name="positron_github",
        title="Positron GitHub Issues and PRs",
    )

    # Generator that yields issues from JSONL
    def issue_generator():
        with open(ISSUES_FILE) as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    store.ingest(issue_generator(), prepare=prepare_issue)

    print("\nBuilding search indexes...")
    store.build_index()

    print(f"\nDone! Store contains {store.size()} documents.")
    print(f"Database saved to: {Path(DB_PATH).absolute()}")


def chat():
    """Start a chat session using chatlas with RAG context."""
    from raghilda.store import DuckDBStore
    from chatlas import ChatOpenAI

    if not Path(DB_PATH).exists():
        print(f"No database found at {DB_PATH}. Run 'download' and 'ingest' first.")
        return

    store = DuckDBStore.connect(location=DB_PATH)

    def retrieve(query: str, top_k: int = 5) -> str:
        """Search the Positron GitHub issues and PRs for relevant information.

        Args:
            query: The search query to find relevant issues, PRs, or discussions.
            top_k: Maximum number of results to return (default: 5).

        Returns:
            Relevant chunks from GitHub issues and PRs matching the query.
        """
        chunks = store.retrieve(query, top_k=top_k)
        results = []
        for chunk in chunks:
            results.append({"text": chunk.text, "context": chunk.context})
        return json.dumps(results)

    chat = ChatOpenAI()
    chat.register_tool(retrieve)
    chat.console()


def main():
    parser = argparse.ArgumentParser(description="Positron GitHub RAG")
    parser.add_argument("commands", nargs="+", choices=["download", "ingest", "chat"])
    args = parser.parse_args()

    for cmd in args.commands:
        if cmd == "download":
            download_issues()
        elif cmd == "ingest":
            ingest_issues()
        elif cmd == "chat":
            chat()


if __name__ == "__main__":
    main()
