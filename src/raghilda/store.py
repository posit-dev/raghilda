from ._store import BaseStore, WriteResult
from ._duckdb_store import DuckDBStore
from ._openai_store import OpenAIStore
from ._chroma_store import ChromaDBStore

__all__ = [
    "BaseStore",
    "WriteResult",
    "DuckDBStore",
    "OpenAIStore",
    "ChromaDBStore",
    "PostgreSQLStore",
]

try:
    from ._sql_store import PostgreSQLStore
except ImportError:
    _pg_import_error: str | None = (
        "PostgreSQLStore requires the 'postgres' extra. "
        "Install it with: pip install raghilda[postgres]"
    )
else:
    _pg_import_error = None


def __getattr__(name: str):
    if name == "PostgreSQLStore" and _pg_import_error is not None:
        raise ImportError(_pg_import_error)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
