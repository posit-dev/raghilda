# /// script
# dependencies = ["PyGithub", "raghilda", "chatlas", "tqdm", "chromadb", "rich<14"]
#
# [tool.uv.sources]
# raghilda = { path = "..", editable = true }
# ///
"""
Example: Building a RAG store from GitHub Issues and PRs

Usage:
    uv run examples/github_issues_rag.py sync posit-dev/positron   # Download & ingest
    uv run examples/github_issues_rag.py chat posit-dev/positron   # Interactive chat
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm


def get_github_token() -> str:
    """Get GitHub token from environment or gh CLI."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token
    result = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True)
    if result.returncode == 0:
        return result.stdout.strip()
    raise RuntimeError("No GitHub token found. Set GITHUB_TOKEN or run 'gh auth login'")


def issue_to_dict(issue) -> dict:
    """Convert a PyGithub issue object to a serializable dict."""
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


def issue_to_document(issue: dict):
    """Convert an issue dict into a chunked MarkdownDocument ready for ingestion.

    This function does three things:
    1. Builds a Markdown string from the issue fields (title, body, comments).
    2. Attaches filterable attributes (state, labels, updated_at) so we can
       narrow searches later (e.g. only open issues, or issues with a label).
    3. Splits the document into smaller chunks using MarkdownChunker. Chunks are
       what actually get embedded and stored — smaller chunks produce more
       focused embeddings. We use a chunk_size of 800 tokens (instead of the
       default 1500) because GitHub issues often contain code blocks, and code
       tokenizes into many more tokens per character than prose.
    """
    from raghilda.chunker import MarkdownChunker
    from raghilda.document import MarkdownDocument

    # -- Step 1: Build a Markdown representation of the issue --
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

    # -- Step 2: Create the document with filterable attributes --
    # Attributes are stored alongside chunks in the database and can be used
    # in filter expressions at query time (e.g. state = 'open').
    updated_at = int(datetime.fromisoformat(issue["updated_at"]).timestamp())
    doc = MarkdownDocument(
        content=content,
        origin=issue["url"],
        attributes={
            "item_number": issue["number"],
            "state": issue["state"],
            "labels": labels if labels else "",
            "updated_at": updated_at,
        },
    )

    # -- Step 3: Split the document into chunks --
    # The chunker respects Markdown structure (headings, code fences, etc.)
    # so chunks stay coherent. Each chunk inherits the document's attributes.
    chunker = MarkdownChunker(chunk_size=800)
    doc = chunker.chunk_document(doc)
    return doc


def sync(repo: str):
    """Download issues from GitHub and build the RAG store (incremental)."""
    from github import Auth, Github
    from raghilda.embedding import EmbeddingOpenAI
    from raghilda.store import ChromaDBStore

    store_path = repo.replace("/", "-") + "_chroma"
    jsonl_path = Path(repo.replace("/", "-") + ".jsonl")
    meta_path = Path(repo.replace("/", "-") + ".meta.json")

    metadata = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    last_update = metadata.get("last_update")

    if Path(store_path).exists() and last_update:
        store = ChromaDBStore.connect("github_issues", location=store_path)
    else:
        store = ChromaDBStore.create(
            location=store_path,
            embed=EmbeddingOpenAI(),
            overwrite=True,
            name="github_issues",
            title=f"GitHub Issues: {repo}",
            attributes={
                "item_number": int,
                "state": str,
                "labels": str,
                "updated_at": int,
            },
        )
        # Seed from JSONL cache if available (e.g. store was deleted)
        if jsonl_path.exists():
            cached_issues = [json.loads(line) for line in open(jsonl_path) if line.strip()]
            store.ingest(cached_issues, prepare=issue_to_document)
            print(f"Rebuilt store from cache ({len(cached_issues)} issues).")

    # Fetch issues from GitHub (incremental if we have a last_update)
    token = get_github_token()
    g = Github(auth=Auth.Token(token))
    github_repo = g.get_repo(repo)

    if last_update:
        print(f"Fetching updates since {last_update}...")
        since = datetime.fromisoformat(last_update)
        issues_iter = github_repo.get_issues(state="all", sort="updated", since=since)
    else:
        print("Fetching all issues...")
        issues_iter = github_repo.get_issues(state="all", sort="updated")

    # Fetch issues: append to JSONL cache and collect for ingestion
    update_time = datetime.now(timezone.utc)
    new_issues = []
    with open(jsonl_path, "a") as f:
        for issue in tqdm(issues_iter, total=issues_iter.totalCount, desc="Fetching"):
            issue_dict = issue_to_dict(issue)
            new_issues.append(issue_dict)
            f.write(json.dumps(issue_dict) + "\n")

    # Ingest only the new/updated issues (ChromaDB upserts by document ID)
    if new_issues:
        print(f"Ingesting {len(new_issues)} issues...")
        store.ingest(new_issues, prepare=issue_to_document)

    # Save sync timestamp so next run only fetches new/updated issues
    metadata["last_update"] = update_time.isoformat()
    meta_path.write_text(json.dumps(metadata, indent=2, default=str))

    print(f"Done! Store contains {store.size()} documents.")


def chat(repo: str):
    """Interactive chat with RAG context from GitHub issues."""
    from chatlas import ChatOpenAI
    from raghilda.store import ChromaDBStore

    store_path = repo.replace("/", "-") + "_chroma"
    if not Path(store_path).exists():
        print(f"No store found at {store_path}. Run 'sync' first.")
        sys.exit(1)

    store = ChromaDBStore.connect("github_issues", location=store_path)

    def retrieve(
        query: str,
        state: str | None = None,
        labels: str | None = None,
        updated_after: str | None = None,
    ) -> str:
        """Search GitHub issues and PRs for relevant information.

        Args:
            query: The search query to find relevant issues/PRs.
            state: Filter by state - "open" or "closed".
            labels: Filter issues that have this label.
            updated_after: Only include items updated after this ISO date (e.g. "2024-01-15").
        """
        filters = []
        if state:
            filters.append({"type": "eq", "key": "state", "value": state})
        if labels:
            filters.append({"type": "eq", "key": "labels", "value": labels})
        if updated_after:
            ts = int(datetime.fromisoformat(updated_after).timestamp())
            filters.append({"type": "gte", "key": "updated_at", "value": ts})

        if len(filters) == 0:
            attributes_filter = None
        elif len(filters) == 1:
            attributes_filter = filters[0]
        else:
            attributes_filter = {"type": "and", "filters": filters}
        chunks = store.retrieve(query, top_k=20, attributes_filter=attributes_filter)

        results = []
        for chunk in chunks:
            result = {"text": chunk.text, "context": chunk.context}
            if hasattr(chunk, "attributes") and chunk.attributes:
                result["attributes"] = chunk.attributes
            results.append(result)
        return json.dumps(results, default=str)

    def current_date() -> str:
        """Return today's date in ISO format (e.g. '2024-06-15')."""
        return datetime.now().strftime("%Y-%m-%d")

    chat_model = ChatOpenAI()
    chat_model.register_tool(retrieve)
    chat_model.register_tool(current_date)
    chat_model.console()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  uv run examples/github_issues_rag.py sync <owner/repo>")
        print("  uv run examples/github_issues_rag.py chat <owner/repo>")
        sys.exit(1)

    command = sys.argv[1]
    repo = sys.argv[2]

    if command == "sync":
        sync(repo)
    elif command == "chat":
        chat(repo)
    else:
        print(f"Unknown command: {command}. Use 'sync' or 'chat'.")
        sys.exit(1)
