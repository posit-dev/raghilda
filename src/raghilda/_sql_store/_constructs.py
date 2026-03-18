"""PostgreSQL-specific SQLAlchemy types and compiled expressions.

Each construct is a FunctionElement subclass compiled via @compiles for
the PostgreSQL dialect. Future backends (Redshift, SQLite, etc.) add their
own @compiles variant without touching the store logic.
"""

from __future__ import annotations

from typing import Any

import sqlalchemy as sa
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.expression import FunctionElement
from sqlalchemy.types import Float


class TSVECTOR(sa.types.UserDefinedType[Any]):
    """Represents PostgreSQL's TSVECTOR type for SQLAlchemy."""

    cache_ok = True

    def get_col_spec(self, **kw: Any) -> str:
        return "TSVECTOR"


class VectorDistance(FunctionElement[Any]):
    """Compute distance between a vector column and a literal vector."""

    inherit_cache = True
    type = Float()
    name = "vector_distance"

    def __init__(self, column: sa.ColumnElement[Any], vector: str, method: str):
        self.column = column
        self.vector = vector
        self.method = method
        super().__init__()


@compiles(VectorDistance, "postgresql")  # type: ignore[misc]
def _pg_vector_distance(element: VectorDistance, compiler: Any, **kw: Any) -> str:
    col_sql = compiler.process(element.column, **kw)
    op_map = {
        "cosine_distance": "<=>",
        "euclidean_distance": "<->",
        "negative_inner_product": "<#>",
    }
    operator = op_map[element.method]
    return f"{col_sql} {operator} '{element.vector}'::vector"


class FTSRank(FunctionElement[Any]):
    """Compute ts_rank of a tsvector column against a text query."""

    inherit_cache = True
    type = Float()
    name = "fts_rank"

    def __init__(self, search_col: sa.ColumnElement[Any], query: str):
        self.search_col = search_col
        self.query = query
        super().__init__()


@compiles(FTSRank, "postgresql")  # type: ignore[misc]
def _pg_fts_rank(element: FTSRank, compiler: Any, **kw: Any) -> str:
    col_sql = compiler.process(element.search_col, **kw)
    bp = sa.literal(element.query)
    bp_sql = compiler.process(bp, **kw)
    return f"ts_rank({col_sql}, plainto_tsquery({bp_sql}))"


class ToSearchVector(FunctionElement[Any]):
    """Convert text to a tsvector for full-text indexing."""

    inherit_cache = True
    name = "to_search_vector"

    def __init__(self, text: str):
        self.text = text
        super().__init__()


@compiles(ToSearchVector, "postgresql")  # type: ignore[misc]
def _pg_to_search_vector(element: ToSearchVector, compiler: Any, **kw: Any) -> str:
    bp = sa.literal(element.text)
    bp_sql = compiler.process(bp, **kw)
    return f"to_tsvector({bp_sql})"


class TextSlice(FunctionElement[Any]):
    """Extract a substring using 1-based start and character length."""

    inherit_cache = True
    name = "text_slice"

    def __init__(
        self,
        text_col: sa.ColumnElement[Any],
        start_expr: sa.ColumnElement[Any],
        length_expr: sa.ColumnElement[Any],
    ):
        self.text_col = text_col
        self.start_expr = start_expr
        self.length_expr = length_expr
        super().__init__()


@compiles(TextSlice, "postgresql")  # type: ignore[misc]
def _pg_text_slice(element: TextSlice, compiler: Any, **kw: Any) -> str:
    text_sql = compiler.process(element.text_col, **kw)
    start_sql = compiler.process(element.start_expr, **kw)
    length_sql = compiler.process(element.length_expr, **kw)
    return f"substring({text_sql} FROM {start_sql} FOR {length_sql})"
