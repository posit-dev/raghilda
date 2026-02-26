import os
import hashlib
import json
import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Any, cast
import duckdb
import httpx
import openai
import pytest
from raghilda.store import DuckDBStore, OpenAIStore
from raghilda.scrape import find_links
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk, RetrievedChunk
from raghilda._attributes import AttributeFloatVectorType
from raghilda._duckdb_store import (
    RetrievedDuckDBMarkdownChunk,
)  # internal implementation
from raghilda._openai_store import _normalize_openai_attributes
from raghilda.embedding import EmbeddingOpenAI
from raghilda._embedding import EmbeddingProvider, EmbedInputType


class CountingEmbedding(EmbeddingProvider):
    def __init__(self):
        self.calls = 0

    def embed(
        self,
        x,
        input_type: EmbedInputType = EmbedInputType.DOCUMENT,
    ):
        self.calls += 1
        return [[float(len(text))] for text in x]

    def get_config(self):
        return {"type": "CountingEmbedding"}

    @classmethod
    def from_config(cls, config):
        return cls()


class _SinglePage:
    def __init__(self, data):
        self.data = data

    def has_next_page(self):
        return False

    def get_next_page(self):
        raise AssertionError("No next page expected")


def _skip_if_unset(env_var: str) -> None:
    if not os.getenv(env_var):
        pytest.skip(f"{env_var} not set in environment variables")


def test_skip_if_unset_skips_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(pytest.skip.Exception):
        _skip_if_unset("OPENAI_API_KEY")


class TestDuckDBStore:
    @pytest.fixture
    def embed(self, request):
        try:
            value = request.param
            if isinstance(value, EmbeddingOpenAI):
                _skip_if_unset("OPENAI_API_KEY")
            return value
        except AttributeError:
            return None

    @pytest.fixture
    def store(self, embed):
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="test_db",
            title="Test DuckDB Store",
        )
        return store

    @pytest.fixture
    def store_with_docs(self, store):
        doc = MarkdownDocument(origin="test", content="This is a test document.")
        doc.chunks = [
            _get_markdown_chunk(doc, start=0, end=4),
            _get_markdown_chunk(doc, start=5, end=7),
            _get_markdown_chunk(doc, start=8, end=9),
            _get_markdown_chunk(doc, start=10, end=14),
            _get_markdown_chunk(doc, start=15, end=23),
        ]
        store.upsert(doc)
        return store

    def test_create_store(self, store):
        assert isinstance(store, DuckDBStore)
        assert store.metadata.name == "test_db"
        assert store.metadata.title == "Test DuckDB Store"
        assert store.metadata.embed is None

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_insert(self, store_with_docs):
        assert store_with_docs.size() == 1

    def test_insert_same_origin_skips_unchanged_by_default(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_skip_unchanged",
        )
        calls_after_create = embed.calls
        doc = MarkdownDocument(origin="doc-1", content="hello world")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]

        first_write = store.upsert(doc)
        assert first_write.document.origin == "doc-1"
        assert first_write.document.content == "hello world"
        assert embed.calls == calls_after_create + 1

        second_write = store.upsert(doc)
        assert second_write.action == "skipped"
        assert second_write.document.origin == "doc-1"
        assert second_write.document.content == "hello world"
        assert embed.calls == calls_after_create + 1

        rows = store.con.execute(
            "SELECT COUNT(*) FROM documents WHERE origin = 'doc-1'"
        ).fetchone()
        assert rows is not None
        assert rows[0] == 1

    def test_insert_same_origin_rewrites_when_skip_disabled(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_force_rewrite",
        )
        calls_after_create = embed.calls
        doc = MarkdownDocument(origin="doc-1", content="hello world")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]

        store.upsert(doc)
        assert embed.calls == calls_after_create + 1

        store.upsert(doc, skip_if_unchanged=False)
        assert embed.calls == calls_after_create + 2

    def test_insert_same_content_but_different_chunking_updates(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_same_content_new_chunks",
        )
        calls_after_create = embed.calls

        content = "hello world"
        doc1 = MarkdownDocument(origin="doc-1", content=content)
        doc1.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content,
                token_count=len(content),
            )
        ]
        first = store.upsert(doc1)
        assert first.action == "inserted"
        assert embed.calls == calls_after_create + 1

        doc2 = MarkdownDocument(origin="doc-1", content=content)
        doc2.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text=content[0:5],
                token_count=5,
            ),
            MarkdownChunk(
                start_index=6,
                end_index=len(content),
                text=content[6:],
                token_count=len(content[6:]),
            ),
        ]
        second = store.upsert(doc2)
        assert second.action == "replaced"
        assert embed.calls == calls_after_create + 2

        chunk_count = store.con.execute(
            "SELECT COUNT(*) FROM embeddings e JOIN documents d USING (doc_id) WHERE d.origin = 'doc-1'"
        ).fetchone()
        assert chunk_count is not None
        assert chunk_count[0] == 2

    def test_insert_same_layout_but_different_chunk_text_updates(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_same_layout_new_text",
        )
        calls_after_create = embed.calls

        content = "hello world"
        doc1 = MarkdownDocument(origin="doc-1", content=content)
        doc1.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content,
                token_count=len(content),
            )
        ]
        first = store.upsert(doc1)
        assert first.action == "inserted"
        assert embed.calls == calls_after_create + 1

        doc2 = MarkdownDocument(origin="doc-1", content=content)
        doc2.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content.upper(),
                token_count=len(content),
            )
        ]
        second = store.upsert(doc2)
        assert second.action == "replaced"
        assert embed.calls == calls_after_create + 2

    def test_insert_same_origin_with_changed_chunk_text_does_not_skip(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_same_origin_chunk_text_change",
        )
        calls_after_create = embed.calls

        content = "Hello World"
        first = MarkdownDocument(origin="doc-1", content=content)
        first.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content.lower(),
                token_count=len(content),
            )
        ]

        inserted = store.upsert(first)
        assert inserted.action == "inserted"
        assert embed.calls == calls_after_create + 1

        second = MarkdownDocument(origin="doc-1", content=content)
        second.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(content),
                text=content,
                token_count=len(content),
            )
        ]

        replaced = store.upsert(second)
        assert replaced.action == "replaced"
        assert embed.calls == calls_after_create + 2

    def test_insert_same_multi_chunk_layout_skips_when_unchanged(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="insert_same_multi_chunk_skip",
            attributes={"tenant": str},
        )
        calls_after_create = embed.calls
        content = "hello world"
        doc = MarkdownDocument(
            origin="doc-1",
            content=content,
            attributes={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text=content[:5],
                token_count=5,
            ),
            MarkdownChunk(
                start_index=6,
                end_index=len(content),
                text=content[6:],
                token_count=len(content[6:]),
            ),
        ]

        first = store.upsert(doc)
        assert first.action == "inserted"
        assert embed.calls == calls_after_create + 1

        second = store.upsert(doc)
        assert second.action == "skipped"
        assert second.document.attributes == {"tenant": "docs"}
        assert embed.calls == calls_after_create + 1

    def test_insert_requires_origin(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="insert_missing_origin",
        )
        doc = MarkdownDocument(origin=None, content="hello world")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]

        with pytest.raises(
            ValueError, match="document.origin must be a non-empty string"
        ):
            store.upsert(doc)

    def test_insert_returns_replaced_document_when_updated(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="insert_replace_snapshot",
            attributes={"tenant": str},
        )
        first = MarkdownDocument(
            origin="doc-1",
            content="hello world",
            attributes={"tenant": "docs"},
        )
        first.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(first.content),
                text=first.content,
                token_count=len(first.content),
            )
        ]

        inserted = store.upsert(first)
        assert inserted.action == "inserted"
        assert inserted.document.origin == "doc-1"
        assert inserted.document.content == "hello world"
        assert inserted.replaced_document is None

        second = MarkdownDocument(
            origin="doc-1",
            content="goodbye world",
            attributes={"tenant": "eng"},
        )
        second.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(second.content),
                text=second.content,
                token_count=len(second.content),
            )
        ]

        updated = store.upsert(second)
        assert updated.action == "replaced"
        assert updated.document.origin == "doc-1"
        assert updated.document.content == "goodbye world"
        assert updated.document.attributes == {"tenant": "eng"}
        assert updated.replaced_document is not None
        assert updated.replaced_document.origin == "doc-1"
        assert updated.replaced_document.content == "hello world"
        assert updated.replaced_document.attributes == {"tenant": "docs"}
        assert updated.replaced_document.chunks is not None
        assert len(updated.replaced_document.chunks) == 1

        restored = store.upsert(updated.replaced_document, skip_if_unchanged=False)
        assert restored.action == "replaced"
        current = store.con.execute(
            "SELECT text FROM documents WHERE origin = 'doc-1'"
        ).fetchone()
        assert current is not None
        assert current[0] == "hello world"

    def test_insert_replaced_document_preserves_multi_chunk_text(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="insert_replace_multi_chunk_snapshot",
        )
        first = MarkdownDocument(origin="doc-1", content="hello world")
        first.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            ),
            MarkdownChunk(
                start_index=6,
                end_index=11,
                text="world",
                token_count=5,
            ),
        ]
        store.upsert(first)

        second = MarkdownDocument(origin="doc-1", content="hello mars")
        second.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            ),
            MarkdownChunk(
                start_index=6,
                end_index=10,
                text="mars",
                token_count=4,
            ),
        ]
        updated = store.upsert(second)
        assert updated.action == "replaced"
        assert updated.replaced_document is not None
        assert updated.replaced_document.chunks is not None
        assert [chunk.text for chunk in updated.replaced_document.chunks] == [
            "hello",
            "world",
        ]

    @pytest.mark.parametrize("embed", [EmbeddingOpenAI()], indirect=True)
    def test_retrieve_vss(self, store_with_docs):
        results = store_with_docs.retrieve_vss("test", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

        results = store_with_docs.retrieve_vss("test", top_k=5)
        assert len(results) == 5

    def test_retrieve_vss_returns_chunk_text_not_document_slice(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            name="vss-text-source",
        )
        doc = MarkdownDocument(origin="vss-text-source", content="alpha beta gamma")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="zeta",
                token_count=4,
            )
        ]
        store.upsert(doc)

        results = store.retrieve_vss([float(len("zeta"))], top_k=1)
        assert len(results) == 1
        assert results[0].text == "zeta"

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_retrieve_bm25(self, store_with_docs):
        store_with_docs.build_index("bm25")
        results = store_with_docs.retrieve_bm25("document", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

    def test_retrieve_bm25_returns_chunk_text_not_document_slice(self, store):
        doc = MarkdownDocument(origin="bm25-text-source", content="alpha beta gamma")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="zeta",
                token_count=4,
            )
        ]
        store.upsert(doc)
        store.build_index("bm25")

        results = store.retrieve_bm25("zeta", top_k=1)
        assert len(results) == 1
        assert results[0].text == "zeta"

    @pytest.mark.parametrize("embed", [EmbeddingOpenAI()], indirect=True)
    def test_retrieve(self, store_with_docs):
        store_with_docs.build_index()
        results = store_with_docs.retrieve("document", top_k=3, deoverlap=False)
        assert len(results) > 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

    def test_retrieve_with_deoverlap(self, store):
        # Create a document with overlapping chunks
        # "hello world test document" = 24 chars
        doc = MarkdownDocument(
            origin="test_deoverlap", content="hello world test document"
        )
        doc.chunks = [
            _get_markdown_chunk(doc, start=0, end=11),  # "hello world"
            _get_markdown_chunk(doc, start=6, end=16),  # "world test"
            _get_markdown_chunk(doc, start=12, end=25),  # "test document"
        ]
        store.upsert(doc)
        store.build_index("bm25")

        # Without deoverlap, we may get multiple overlapping chunks
        results_no_deoverlap = store.retrieve("test", top_k=5, deoverlap=False)

        # With deoverlap, overlapping chunks should be merged
        results_deoverlap = store.retrieve("test", top_k=5, deoverlap=True)

        # Deoverlapped results should have fewer or equal chunks
        assert len(results_deoverlap) <= len(results_no_deoverlap)

        # Check that deoverlapped chunks don't have overlapping ranges
        for i, chunk1 in enumerate(results_deoverlap):
            for chunk2 in results_deoverlap[i + 1 :]:
                if chunk1.doc_id == chunk2.doc_id:
                    # Same document - ranges should not overlap
                    assert (
                        chunk1.end_index <= chunk2.start_index
                        or chunk2.end_index <= chunk1.start_index
                    ), (
                        f"Chunks overlap: [{chunk1.start_index}, {chunk1.end_index}) and [{chunk2.start_index}, {chunk2.end_index})"
                    )

    def test_retrieve_with_deoverlap_aggregates_attributes(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"topic": str},
        )
        doc = MarkdownDocument(
            origin="test_deoverlap_attributes",
            content="alpha beta gamma",
            attributes={"topic": "first"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=10,
                text="alpha beta",
                token_count=10,
                context="h1",
            ),
            MarkdownChunk(
                start_index=6,
                end_index=16,
                text="beta gamma",
                token_count=10,
                context="h2",
                attributes={"topic": "second"},
            ),
        ]
        store.upsert(doc)
        store.build_index("bm25")

        results = store.retrieve("beta", top_k=5, deoverlap=True)

        assert len(results) == 1
        assert results[0].context == "h1"
        assert results[0].attributes == {"topic": ["first", "second"]}

    def test_retrieve_supports_excluding_seen_chunk_ids(self, store):
        content = "alpha one alpha two alpha three"
        doc = MarkdownDocument(origin="chunk-id-filter", content=content)
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=9,
                text=content[0:9],
                token_count=9,
            ),
            MarkdownChunk(
                start_index=10,
                end_index=19,
                text=content[10:19],
                token_count=9,
            ),
            MarkdownChunk(
                start_index=20,
                end_index=31,
                text=content[20:31],
                token_count=11,
            ),
        ]
        store.upsert(doc)
        store.build_index("bm25")

        first_results = store.retrieve("alpha", top_k=2, deoverlap=False)
        seen_chunk_ids = [
            chunk_id for chunk in first_results for chunk_id in chunk.chunk_ids
        ]
        assert len(seen_chunk_ids) == 2

        remaining_results = store.retrieve(
            "alpha",
            top_k=3,
            deoverlap=False,
            attributes_filter={
                "type": "nin",
                "key": "chunk_id",
                "value": seen_chunk_ids,
            },
        )
        remaining_chunk_ids = {
            chunk_id for chunk in remaining_results for chunk_id in chunk.chunk_ids
        }
        assert remaining_chunk_ids
        assert remaining_chunk_ids.isdisjoint(set(seen_chunk_ids))

    def test_create_store_with_attributes_schema(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="attributes_schema_db",
            title="Attributes Schema Store",
            attributes={
                "tenant": str,
                "priority": int,
                "is_public": bool,
            },
        )

        assert store.metadata.attributes_schema == {
            "tenant": str,
            "priority": int,
            "is_public": bool,
        }

        columns = store.con.execute("DESCRIBE embeddings").fetchall()
        columns_by_name = {row[0]: row[1] for row in columns}
        names = set(columns_by_name)
        assert "tenant" in names
        assert "priority" in names
        assert "is_public" in names
        assert columns_by_name["tenant"] == "VARCHAR"
        assert columns_by_name["priority"] == "INTEGER"
        assert columns_by_name["is_public"] == "BOOLEAN"

    def test_create_store_with_attributes_schema_class_annotations(self):
        class AttributesSpec:
            tenant: str
            priority: int
            is_public: bool

        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes=AttributesSpec,
        )

        assert store.metadata.attributes_schema == {
            "tenant": str,
            "priority": int,
            "is_public": bool,
        }

    def test_create_store_rejects_invalid_attribute_names(self):
        with pytest.raises(ValueError, match="must match"):
            DuckDBStore.create(
                location=":memory:",
                embed=None,
                overwrite=True,
                attributes={"tenant-id": str},
            )

    def test_create_store_with_vector_attributes_annotation(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "embedding25": Annotated[list[float], 25],
            },
        )

        vector_type = store.metadata.attributes_schema["embedding25"]
        assert isinstance(vector_type, AttributeFloatVectorType)
        assert vector_type.dimension == 25

        columns = store.con.execute("DESCRIBE embeddings").fetchall()
        columns_by_name = {row[0]: row[1] for row in columns}
        assert columns_by_name["embedding25"] == "FLOAT[25]"

        vector = [float(i) for i in range(25)]
        doc = MarkdownDocument(
            origin="vector-attributes",
            content="hello vector attributes",
            attributes={"tenant": "docs", "embedding25": vector},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]
        store.upsert(doc)
        store.build_index("bm25")

        results = store.retrieve(
            "hello",
            top_k=1,
            deoverlap=False,
        )
        assert len(results) == 1
        assert results[0].attributes is not None
        assert results[0].attributes["embedding25"] == pytest.approx(vector)

        filterable_columns = store._filterable_columns()
        assert "tenant" in filterable_columns
        assert "start_index" in filterable_columns
        assert "end_index" in filterable_columns
        assert "start" not in filterable_columns
        assert "end" not in filterable_columns
        assert "embedding25" not in filterable_columns

        with pytest.raises(ValueError, match="Unknown attribute column 'embedding25'"):
            store.retrieve(
                "hello",
                top_k=1,
                deoverlap=False,
                attributes_filter="embedding25 = 1",
            )

    def test_insert_same_vector_attributes_skips_when_unchanged(self):
        embed = CountingEmbedding()
        store = DuckDBStore.create(
            location=":memory:",
            embed=embed,
            overwrite=True,
            attributes={
                "tenant": str,
                "embedding3": Annotated[list[float], 3],
            },
        )
        calls_after_create = embed.calls

        vector = [1.0, 2.0, 3.0]
        doc = MarkdownDocument(
            origin="vector-unchanged",
            content="hello vector unchanged",
            attributes={"tenant": "docs", "embedding3": vector},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(doc.content),
                text=doc.content,
                token_count=len(doc.content),
            )
        ]

        first = store.upsert(doc)
        assert first.action == "inserted"
        assert embed.calls == calls_after_create + 1

        second = store.upsert(doc)
        assert second.action == "skipped"
        assert embed.calls == calls_after_create + 1

    def test_insert_and_retrieve_with_attributes_filter(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "priority": int,
                "is_public": bool | None,
            },
        )

        doc = MarkdownDocument(
            origin="attributes-test",
            content="alpha beta gamma",
            attributes={"tenant": "docs", "priority": 1},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
                attributes={"priority": 5, "is_public": False},
            ),
            MarkdownChunk(
                start_index=6,
                end_index=10,
                text="beta",
                token_count=4,
            ),
            MarkdownChunk(
                start_index=11,
                end_index=16,
                text="gamma",
                token_count=5,
            ),
        ]

        store.upsert(doc)
        store.build_index("bm25")

        private_results = store.retrieve(
            "alpha",
            top_k=10,
            deoverlap=False,
            attributes_filter="tenant = 'docs' AND priority >= 5",
        )
        assert len(private_results) == 1
        assert private_results[0].text.strip() == "alpha"
        assert private_results[0].attributes == {
            "tenant": "docs",
            "priority": 5,
            "is_public": False,
        }

        public_results = store.retrieve(
            "beta",
            top_k=10,
            deoverlap=False,
            attributes_filter="tenant = 'docs' AND is_public IS NULL AND priority = 1",
        )
        assert len(public_results) == 1
        assert public_results[0].text.strip() == "beta"
        assert public_results[0].attributes == {
            "tenant": "docs",
            "priority": 1,
            "is_public": None,
        }

        dict_results = store.retrieve(
            "alpha",
            top_k=10,
            deoverlap=False,
            attributes_filter={
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "tenant", "value": "docs"},
                    {"type": "in", "key": "priority", "value": [5, 10]},
                ],
            },
        )
        assert len(dict_results) == 1
        assert dict_results[0].text.strip() == "alpha"

        positional_results = store.retrieve(
            "alpha",
            top_k=10,
            deoverlap=False,
            attributes_filter="start_index = 0 AND end_index = 5",
        )
        assert len(positional_results) == 1
        assert positional_results[0].text.strip() == "alpha"

        with pytest.raises(ValueError, match="Unknown attribute column 'start'"):
            store.retrieve(
                "alpha",
                top_k=10,
                deoverlap=False,
                attributes_filter="start = 0",
            )

    def test_insert_preserves_declared_id_attribute(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "id": str,
                "tenant": str,
            },
        )

        doc = MarkdownDocument(
            origin="id-attribute-test",
            content="alpha beta",
            attributes={"id": "attr-id-1", "tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        store.upsert(doc)
        store.build_index("bm25")

        results = store.retrieve(
            "alpha",
            top_k=5,
            deoverlap=False,
            attributes_filter="id = 'attr-id-1'",
        )
        assert len(results) == 1
        assert results[0].attributes == {"id": "attr-id-1", "tenant": "docs"}

    def test_retrieve_rejects_text_attribute_filter(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
        )

        doc = MarkdownDocument(
            origin="text-filter-test",
            content="alpha beta",
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        store.upsert(doc)
        store.build_index("bm25")

        with pytest.raises(ValueError, match="Unknown attribute column 'text'"):
            store.retrieve(
                "alpha",
                top_k=5,
                deoverlap=False,
                attributes_filter="text = 'alpha'",
            )

    def test_insert_and_retrieve_with_nested_object_attributes_filter(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "details": {
                    "source": str,
                    "flags": {"is_public": bool, "is_internal": bool},
                },
            },
        )

        doc = MarkdownDocument(
            origin="nested-attributes-test",
            content="alpha beta gamma",
            attributes={
                "tenant": "docs",
                "details": {
                    "source": "handbook",
                    "flags": {"is_public": True, "is_internal": False},
                },
            },
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        store.upsert(doc)
        store.build_index("bm25")

        results = store.retrieve(
            "alpha",
            top_k=5,
            deoverlap=False,
            attributes_filter=(
                "tenant = 'docs' "
                "AND details.source = 'handbook' "
                "AND details.flags.is_public = TRUE"
            ),
        )
        assert len(results) == 1
        assert results[0].text.strip() == "alpha"
        assert results[0].attributes == {
            "tenant": "docs",
            "details": {
                "source": "handbook",
                "flags": {"is_public": True, "is_internal": False},
            },
        }

    def test_nested_object_attribute_rejects_hyphenated_field_name(self):
        with pytest.raises(ValueError, match="must match"):
            DuckDBStore.create(
                location=":memory:",
                embed=None,
                overwrite=True,
                attributes={
                    "details": {
                        "source-type": str,
                    },
                },
            )

    def test_insert_applies_inline_attribute_defaults(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "priority": (int, 0),
                "is_public": (bool, False),
                "topic": (str | None, None),
            },
        )

        doc = MarkdownDocument(
            origin="defaults-test",
            content="alpha beta",
            attributes={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        store.upsert(doc)
        store.build_index("bm25")

        results = store.retrieve("alpha", top_k=5, deoverlap=False)
        assert len(results) == 1
        assert results[0].attributes == {
            "tenant": "docs",
            "priority": 0,
            "is_public": False,
            "topic": None,
        }

    def test_insert_result_document_preserves_defaulted_attributes(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "priority": (int, 0),
            },
        )

        doc = MarkdownDocument(
            origin="defaults-in-result",
            content="alpha",
            attributes={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        inserted = store.upsert(doc)
        assert inserted.action == "inserted"
        assert inserted.document.attributes == {
            "tenant": "docs",
            "priority": 0,
        }

        updated = MarkdownDocument(
            origin="defaults-in-result",
            content="alpha beta",
            attributes={"tenant": "docs"},
        )
        updated.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=10,
                text="alpha beta",
                token_count=10,
            )
        ]

        replaced = store.upsert(updated, skip_if_unchanged=False)
        assert replaced.action == "replaced"
        assert replaced.document.attributes == {
            "tenant": "docs",
            "priority": 0,
        }

    def test_insert_snapshot_reads_are_serialized_under_db_lock(self, monkeypatch):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"tenant": str, "priority": (int, 0)},
        )

        observed_lock_states: list[bool] = []
        original_snapshot = store._load_document_snapshot

        def wrapped_snapshot(*, doc_id: str, origin: str, text: str):
            observed_lock_states.append(store._db_lock.locked())
            return original_snapshot(doc_id=doc_id, origin=origin, text=text)

        monkeypatch.setattr(store, "_load_document_snapshot", wrapped_snapshot)

        first = MarkdownDocument(
            origin="lock-snapshot-test",
            content="alpha",
            attributes={"tenant": "docs"},
        )
        first.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]
        store.upsert(first, skip_if_unchanged=False)

        second = MarkdownDocument(
            origin="lock-snapshot-test",
            content="alpha beta",
            attributes={"tenant": "docs"},
        )
        second.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=10,
                text="alpha beta",
                token_count=10,
            )
        ]
        store.upsert(second, skip_if_unchanged=False)

        assert observed_lock_states
        assert all(observed_lock_states)

    def test_insert_snapshot_preserves_nullable_none_attributes(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={
                "tenant": str,
                "topic": (str | None, "general"),
            },
        )

        first = MarkdownDocument(
            origin="nullable-snapshot-test",
            content="alpha",
            attributes={"tenant": "docs", "topic": None},
        )
        first.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="alpha",
                token_count=5,
            )
        ]

        inserted = store.upsert(first)
        assert inserted.action == "inserted"
        assert inserted.document.attributes == {
            "tenant": "docs",
            "topic": None,
        }

        second = MarkdownDocument(
            origin="nullable-snapshot-test",
            content="alpha beta",
            attributes={"tenant": "docs", "topic": "updated"},
        )
        second.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=10,
                text="alpha beta",
                token_count=10,
            )
        ]

        replaced = store.upsert(second, skip_if_unchanged=False)
        assert replaced.action == "replaced"
        assert replaced.replaced_document is not None
        assert replaced.replaced_document.attributes == {
            "tenant": "docs",
            "topic": None,
        }

        restored = store.upsert(replaced.replaced_document, skip_if_unchanged=False)
        assert restored.action == "replaced"
        assert restored.document.attributes == {
            "tenant": "docs",
            "topic": None,
        }

    def test_insert_replaced_snapshot_uses_existing_doc_id_for_legacy_rows(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"tenant": str},
        )

        legacy_doc_id = "legacy-doc-id"
        store.con.execute(
            "INSERT INTO documents (doc_id, origin, text) VALUES (?, ?, ?)",
            [legacy_doc_id, "legacy-origin", "alpha"],
        )
        store.con.execute(
            """
            INSERT INTO embeddings (
                doc_id,
                start_index,
                end_index,
                chunk_text,
                context,
                tenant
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [legacy_doc_id, 0, 5, "alpha", None, "old"],
        )

        updated = MarkdownDocument(
            origin="legacy-origin",
            content="alpha beta",
            attributes={"tenant": "new"},
        )
        updated.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=10,
                text="alpha beta",
                token_count=10,
            )
        ]

        replaced = store.upsert(updated, skip_if_unchanged=False)
        assert replaced.action == "replaced"
        assert replaced.document.chunks is not None
        assert [chunk.text for chunk in replaced.document.chunks] == ["alpha beta"]
        assert replaced.document.attributes == {"tenant": "new"}
        assert replaced.replaced_document is not None
        assert [chunk.text for chunk in replaced.replaced_document.chunks or []] == [
            "alpha"
        ]
        assert replaced.replaced_document.attributes == {"tenant": "old"}

    def test_insert_missing_required_attribute_fails(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"tenant": str, "priority": (int, 0)},
        )

        doc = MarkdownDocument(
            origin="required-fail",
            content="hello",
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(ValueError, match="Missing required attribute 'tenant'"):
            store.upsert(doc)

    def test_insert_attributes_without_declared_schema_fails(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
        )

        doc = MarkdownDocument(
            origin="attributes-fail",
            content="hello",
            attributes={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(ValueError, match="Unknown attribute key 'tenant'"):
            store.upsert(doc)

    def test_insert_unknown_chunk_attributes_key_fails(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"tenant": str},
        )

        doc = MarkdownDocument(
            origin="unknown-key-fail",
            content="hello",
            attributes={"tenant": "docs"},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
                attributes={"unknown": "x"},
            )
        ]

        with pytest.raises(ValueError, match="Unknown attribute key 'unknown'"):
            store.upsert(doc)

    def test_insert_rejects_float_for_int_attribute(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"priority": int},
        )

        doc = MarkdownDocument(
            origin="type-mismatch-float-for-int",
            content="hello",
            attributes={"priority": 1.5},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(
            ValueError,
            match="Invalid value for attributes 'priority': expected int, got float",
        ):
            store.upsert(doc)

    def test_insert_rejects_int_for_float_attribute(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            attributes={"score": float},
        )

        doc = MarkdownDocument(
            origin="type-mismatch-int-for-float",
            content="hello",
            attributes={"score": 1},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]

        with pytest.raises(
            ValueError,
            match="Invalid value for attributes 'score': expected float, got int",
        ):
            store.upsert(doc)

    def test_connect_restores_attributes_schema(self, tmp_path):
        db_path = tmp_path / "attributes-connect.db"
        store = DuckDBStore.create(
            location=str(db_path),
            embed=None,
            overwrite=True,
            attributes={"tenant": str, "priority": int},
        )
        store.con.close()

        store2 = DuckDBStore.connect(str(db_path))
        assert store2.metadata.attributes_schema == {
            "tenant": str,
            "priority": int,
        }


class TestOpenAIStore:
    @pytest.fixture(autouse=True)
    def setup(self):
        _skip_if_unset("OPENAI_API_KEY")

    @pytest.fixture
    def store(self):
        store = OpenAIStore.create()
        try:
            yield store
        finally:
            try:
                store.client.vector_stores.delete(vector_store_id=store.store_id)
            except openai.AuthenticationError:
                pass

    @pytest.fixture(scope="class")
    def store_with_attributes(self):
        _skip_if_unset("OPENAI_API_KEY")
        store = OpenAIStore.create(attributes={"tenant": str, "priority": int})
        store.upsert(
            MarkdownDocument(
                origin="doc-attrs",
                content="alpha bronze owl",
                attributes={"tenant": "docs", "priority": 2},
            ),
        )
        store.upsert(
            MarkdownDocument(
                origin="docs-priority-1",
                content="alpha beta",
                attributes={"tenant": "docs", "priority": 1},
            ),
        )
        store.upsert(
            MarkdownDocument(
                origin="ops-priority-5",
                content="alpha gamma",
                attributes={"tenant": "ops", "priority": 5},
            ),
        )
        store.upsert(
            MarkdownDocument(
                origin="docs-priority-3",
                content="alpha alpha delta",
                attributes={"tenant": "docs", "priority": 3},
            ),
        )
        try:
            yield store
        finally:
            try:
                store.client.vector_stores.delete(vector_store_id=store.store_id)
            except openai.AuthenticationError:
                pass

    @pytest.fixture
    def store_with_class_attributes(self):
        class AttributesSpec:
            tenant: str
            priority: int

        _skip_if_unset("OPENAI_API_KEY")
        store = OpenAIStore.create(attributes=AttributesSpec)
        try:
            yield store
        finally:
            try:
                store.client.vector_stores.delete(vector_store_id=store.store_id)
            except openai.AuthenticationError:
                pass

    @pytest.fixture
    def store_with_docs(self, store):
        doc = MarkdownDocument(
            origin="test", content="hello world this is a document world world world"
        )
        store.upsert(doc)
        return store

    def test_create_store(self, store):
        assert isinstance(store, OpenAIStore)
        assert isinstance(store.store_id, str)

    def test_insert(self, store_with_docs):
        assert store_with_docs.size() == 1

    def test_retrieve(self, store_with_docs):
        results = store_with_docs.retrieve("world", top_k=3)
        assert len(results) > 0
        for chunk in results:
            assert isinstance(chunk, RetrievedChunk)
            assert chunk.text is not None

    def test_connect_restores_attributes_schema(self, store_with_attributes):
        connected = OpenAIStore.connect(store_id=store_with_attributes.store_id)
        assert connected.attributes_schema == {"tenant": str, "priority": int}

    def test_create_accepts_class_attributes_schema(self, store_with_class_attributes):
        connected = OpenAIStore.connect(store_id=store_with_class_attributes.store_id)
        assert connected.attributes_schema == {"tenant": str, "priority": int}

    def test_insert_uses_document_attributes(self, store_with_attributes):
        results = store_with_attributes.retrieve(
            "bronze owl",
            top_k=5,
            attributes_filter="tenant = 'docs' AND priority = 2",
        )
        assert len(results) > 0
        assert all(chunk.attributes["tenant"] == "docs" for chunk in results)
        assert all(float(chunk.attributes["priority"]) == 2.0 for chunk in results)

    def test_retrieve_supports_attributes_filter(self, store_with_attributes):
        results = store_with_attributes.retrieve(
            "alpha",
            top_k=5,
            attributes_filter="tenant = 'docs' AND priority >= 2",
        )

        assert len(results) > 0
        assert all(chunk.attributes["tenant"] == "docs" for chunk in results)
        assert all(float(chunk.attributes["priority"]) >= 2.0 for chunk in results)

    def test_retrieve_supports_attributes_filter_ast(self, store_with_attributes):
        results = store_with_attributes.retrieve(
            "alpha",
            top_k=5,
            attributes_filter={
                "type": "and",
                "filters": [
                    {"type": "eq", "key": "tenant", "value": "docs"},
                    {"type": "in", "key": "priority", "value": [2, 3]},
                ],
            },
        )
        assert len(results) > 0
        assert all(chunk.attributes["tenant"] == "docs" for chunk in results)
        assert all(
            float(chunk.attributes["priority"]) in (2.0, 3.0) for chunk in results
        )

    def test_rejects_chunked_documents(self, store_with_attributes):
        doc = MarkdownDocument(
            origin="chunk-attrs",
            content="hello",
            attributes={"tenant": "docs", "priority": 1},
        )
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
                attributes={"tenant": "docs"},
            )
        ]

        with pytest.raises(
            ValueError, match="OpenAIStore does not support chunked documents"
        ):
            store_with_attributes.upsert(doc)


def test_openai_store_create_rejects_vector_attributes_schema():
    with pytest.raises(ValueError, match="Vector attribute types are not supported"):
        OpenAIStore.create(attributes={"embedding25": Annotated[list[float], 25]})


def test_openai_store_create_accepts_mapping_attributes_with_mocked_client(monkeypatch):
    class FakeVectorStores:
        def create(self, **kwargs):
            return SimpleNamespace(id="vs_test")

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    store = OpenAIStore.create(attributes={"tenant": str, "priority": int})
    assert store.attributes_schema == {"tenant": str, "priority": int}
    assert set(store.attributes_spec.keys()) == {"tenant", "priority"}


def test_openai_store_normalize_attributes_preserves_large_ints():
    normalized = _normalize_openai_attributes(
        {
            "tenant_id": 9007199254740992,
            "doc_id": 9007199254740993,
            "score": 0.75,
            "active": True,
            "label": "docs",
        }
    )
    assert normalized == {
        "tenant_id": 9007199254740992,
        "doc_id": 9007199254740993,
        "score": 0.75,
        "active": True,
        "label": "docs",
    }
    assert isinstance(normalized["tenant_id"], int)
    assert isinstance(normalized["doc_id"], int)
    assert isinstance(normalized["score"], float)


def test_openai_store_create_rejects_object_attributes_schema():
    with pytest.raises(ValueError, match="Object attribute types are not supported"):
        OpenAIStore.create(attributes={"details": {"source": str}})


def test_openai_store_create_rejects_optional_attributes_schema():
    with pytest.raises(
        ValueError, match="Optional attribute values are not supported for 'topic'"
    ):
        OpenAIStore.create(attributes={"topic": str | None})


def test_openai_store_create_rejects_defaulted_attributes_schema():
    with pytest.raises(
        ValueError, match="Optional attribute values are not supported for 'priority'"
    ):
        OpenAIStore.create(attributes={"tenant": str, "priority": (int, 0)})


def test_openai_store_create_rejects_invalid_attribute_names():
    with pytest.raises(ValueError, match="must match"):
        OpenAIStore.create(attributes={"tenant-id": str})


def test_openai_store_create_rejects_more_than_14_user_attributes(monkeypatch):
    class FakeVectorStores:
        def create(self, **kwargs):
            raise AssertionError("create should not be called for invalid schema")

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    with pytest.raises(ValueError, match="at most 14 user attributes"):
        OpenAIStore.create(attributes={f"k{i}": str for i in range(15)})


@pytest.mark.parametrize(
    "attribute_name",
    ["_raghilda_origin", "_raghilda_content_hash"],
)
def test_openai_store_rejects_internal_attribute_names(attribute_name):
    with pytest.raises(ValueError, match=attribute_name):
        OpenAIStore(
            client=SimpleNamespace(),
            store_id="vs_test",
            attributes={attribute_name: str},
        )


def test_openai_store_insert_updates_when_attributes_change_for_same_content():
    content = "hello world"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": content_hash,
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_old"
            return SimpleNamespace(content=content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )
    assert result.action == "replaced"
    assert result.document.origin == "doc"
    assert result.replaced_document is not None
    assert result.replaced_document.origin == "doc"
    assert result.replaced_document.content == content
    assert fake_vector_store_files.deleted_ids == ["file_old"]
    assert len(fake_vector_store_files.upload_calls) == 1
    assert fake_vector_store_files.upload_calls[0]["attributes"]["tenant"] == "new"


def test_openai_store_insert_unchanged_returns_stable_snapshot_id():
    content = "hello world"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": content_hash,
                            "tenant": "new",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_old"
            return SimpleNamespace(content=content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    first = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )
    second = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )

    assert first.action == "skipped"
    assert second.action == "skipped"
    assert first.document.origin == "doc"
    assert second.document.origin == "doc"
    assert fake_vector_store_files.upload_calls == []
    assert fake_vector_store_files.deleted_ids == []


def test_openai_store_insert_skipped_fallback_preserves_existing_file_id():
    content = "hello world"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": content_hash,
                            "tenant": "new",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            request = httpx.Request(
                "GET", f"https://api.openai.com/v1/files/{file_id}/content"
            )
            raise openai.APIConnectionError(request=request)

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "skipped"
    assert result.document.origin == "doc"
    assert fake_vector_store_files.upload_calls == []
    assert fake_vector_store_files.deleted_ids == []


def test_openai_store_insert_handles_multiple_managed_files_for_origin():
    content = "hello world"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_one",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": "hash-a",
                            "tenant": "new",
                        },
                    ),
                    SimpleNamespace(
                        id="file_two",
                        created_at=2,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": content_hash,
                            "tenant": "new",
                        },
                    ),
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_two"
            return SimpleNamespace(content=content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "skipped"
    assert result.document.origin == "doc"
    assert fake_vector_store_files.upload_calls == []
    assert fake_vector_store_files.deleted_ids == ["file_one"]


def test_openai_store_insert_skipped_when_duplicate_cleanup_delete_fails():
    content = "hello world"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.delete_calls = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_one",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": "hash-a",
                            "tenant": "new",
                        },
                    ),
                    SimpleNamespace(
                        id="file_two",
                        created_at=2,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": content_hash,
                            "tenant": "new",
                        },
                    ),
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.delete_calls.append(file_id)
            raise RuntimeError("temporary delete failure")

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_two"
            return SimpleNamespace(content=content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "skipped"
    assert result.document.origin == "doc"
    assert fake_vector_store_files.upload_calls == []
    assert fake_vector_store_files.delete_calls == ["file_one"]


def test_openai_store_insert_returns_uploaded_file_id_when_new_document():
    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage([])

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_uploaded")

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=SimpleNamespace(content=lambda file_id: None),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content="hello world",
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "inserted"
    assert result.document.origin == "doc"
    assert result.replaced_document is None
    assert len(fake_vector_store_files.upload_calls) == 1
    assert fake_vector_store_files.deleted_ids == []


def test_openai_store_insert_rejects_too_many_user_attributes():
    class FakeVectorStoreFiles:
        def __init__(self):
            self.upload_calls = []
            self.page = _SinglePage([])

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            raise AssertionError("delete should not be called")

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=SimpleNamespace(content=lambda file_id: None),
    )
    attributes_schema = {f"k{i}": str for i in range(15)}
    with pytest.raises(ValueError, match="at most 14 user attributes"):
        OpenAIStore(
            client=fake_client,
            store_id="vs_test",
            attributes=attributes_schema,
        )
    assert fake_vector_store_files.upload_calls == []


def test_openai_store_insert_ignores_matching_filename_without_internal_origin():
    content = "hello world"

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_existing",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=SimpleNamespace(content=lambda file_id: SimpleNamespace(content=b"")),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )
    assert result.action == "inserted"
    assert result.replaced_document is None
    assert fake_vector_store_files.deleted_ids == []
    assert len(fake_vector_store_files.upload_calls) == 1
    assert fake_vector_store_files.upload_calls[0]["attributes"]["tenant"] == "new"


def test_openai_store_insert_ignores_unmanaged_matching_filename():
    content = "hello world"

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_unmanaged",
                        created_at=1,
                        filename="doc.md",
                        attributes={"source": "external"},
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=SimpleNamespace(content=lambda file_id: SimpleNamespace(content=b"")),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "inserted"
    assert result.replaced_document is None
    assert fake_vector_store_files.deleted_ids == []
    assert len(fake_vector_store_files.upload_calls) == 1


def test_openai_store_insert_keeps_existing_file_when_upload_fails():
    content = "hello world"
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": content_hash,
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            raise RuntimeError("upload failed")

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_old"
            return SimpleNamespace(content=content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    with pytest.raises(RuntimeError, match="upload failed"):
        store.upsert(
            MarkdownDocument(
                origin="doc",
                content="new content",
                attributes={"tenant": "new"},
            )
        )
    assert fake_vector_store_files.deleted_ids == []


def test_openai_store_insert_serializes_replacement_for_same_origin():
    import threading

    old_content = "hello world"
    old_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()
    first_two_list_calls = threading.Barrier(2)
    state_lock = threading.Lock()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.files = [
                {
                    "id": "file_old",
                    "created_at": 1,
                    "filename": "doc.md",
                    "attributes": {
                        "_raghilda_origin": "doc",
                        "_raghilda_content_hash": old_hash,
                        "tenant": "old",
                    },
                }
            ]
            self._next_id = 2
            self._list_calls = 0

        def list(self, **kwargs):
            with state_lock:
                self._list_calls += 1
                snapshot = [
                    SimpleNamespace(
                        id=file["id"],
                        created_at=file["created_at"],
                        filename=file["filename"],
                        attributes=dict(file["attributes"]),
                    )
                    for file in self.files
                ]
            if self._list_calls <= 2:
                try:
                    first_two_list_calls.wait(timeout=0.2)
                except threading.BrokenBarrierError:
                    pass
            return _SinglePage(snapshot)

        def delete(self, file_id, **kwargs):
            with state_lock:
                self.deleted_ids.append(file_id)
                self.files = [file for file in self.files if file["id"] != file_id]
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            with state_lock:
                file_id = f"file_new_{self._next_id}"
                self._next_id += 1
                self.files.append(
                    {
                        "id": file_id,
                        "created_at": self._next_id,
                        "filename": kwargs["file"][0],
                        "attributes": dict(kwargs.get("attributes", {})),
                    }
                )
                self.upload_calls.append(kwargs)
            return SimpleNamespace(id=file_id)

    class FakeFiles:
        def content(self, file_id):
            return SimpleNamespace(content=old_content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    errors: list[Exception] = []

    def do_insert(content: str):
        try:
            store.upsert(
                MarkdownDocument(
                    origin="doc",
                    content=content,
                    attributes={"tenant": "new"},
                ),
                skip_if_unchanged=False,
            )
        except Exception as error:
            errors.append(error)

    t1 = threading.Thread(target=do_insert, args=("content thread one",))
    t2 = threading.Thread(target=do_insert, args=("content thread two",))
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert errors == []
    managed = [
        file
        for file in fake_vector_store_files.files
        if file["attributes"].get("_raghilda_origin") == "doc"
    ]
    assert len(managed) == 1


def test_openai_store_insert_raises_when_old_file_delete_fails():
    old_content = "hello world"
    new_content = "hello world updated"
    old_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": old_hash,
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            request = httpx.Request(
                "DELETE",
                f"https://api.openai.com/v1/vector_stores/{kwargs.get('vector_store_id')}/files/{file_id}",
            )
            raise openai.APIConnectionError(request=request)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_old"
            return SimpleNamespace(content=old_content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    with pytest.raises(openai.APIConnectionError):
        store.upsert(
            MarkdownDocument(
                origin="doc",
                content=new_content,
                attributes={"tenant": "new"},
            )
        )
    assert fake_vector_store_files.deleted_ids == ["file_old", "file_new"]
    assert len(fake_vector_store_files.upload_calls) == 1


def test_openai_store_insert_rolls_back_uploaded_file_when_old_file_delete_fails():
    old_content = "hello world"
    new_content = "hello world updated"
    old_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self._next_id = 1
            self._old_delete_failed_once = False
            self.files = [
                {
                    "id": "file_old",
                    "created_at": 1,
                    "filename": "doc.md",
                    "attributes": {
                        "_raghilda_origin": "doc",
                        "_raghilda_content_hash": old_hash,
                        "tenant": "old",
                    },
                }
            ]

        def list(self, **kwargs):
            snapshot = [
                SimpleNamespace(
                    id=file["id"],
                    created_at=file["created_at"],
                    filename=file["filename"],
                    attributes=dict(file["attributes"]),
                )
                for file in self.files
            ]
            return _SinglePage(snapshot)

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            if file_id == "file_old" and not self._old_delete_failed_once:
                self._old_delete_failed_once = True
                request = httpx.Request(
                    "DELETE",
                    f"https://api.openai.com/v1/vector_stores/{kwargs.get('vector_store_id')}/files/{file_id}",
                )
                raise openai.APIConnectionError(request=request)

            self.files = [file for file in self.files if file["id"] != file_id]
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            file_id = f"file_new_{self._next_id}"
            self._next_id += 1
            self.files.append(
                {
                    "id": file_id,
                    "created_at": self._next_id,
                    "filename": kwargs["file"][0],
                    "attributes": dict(kwargs.get("attributes", {})),
                }
            )
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id=file_id)

    class FakeFiles:
        def content(self, file_id):
            assert file_id == "file_old"
            return SimpleNamespace(content=old_content.encode("utf-8"))

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    with pytest.raises(openai.APIConnectionError):
        store.upsert(
            MarkdownDocument(
                origin="doc",
                content=new_content,
                attributes={"tenant": "new"},
            )
        )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=new_content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "replaced"
    managed_doc_files = [
        file
        for file in fake_vector_store_files.files
        if file["attributes"].get("_raghilda_origin") == "doc"
    ]
    assert len(managed_doc_files) == 1
    assert managed_doc_files[0]["id"] == "file_new_2"
    assert fake_vector_store_files.deleted_ids == ["file_old", "file_new_1", "file_old"]
    assert len(fake_vector_store_files.upload_calls) == 2


def test_openai_store_connect_restores_metadata_schema_with_mocked_client(
    monkeypatch,
):
    schema_json = json.dumps(
        {
            "tenant": {"type": "str", "nullable": False, "required": True},
            "priority": {"type": "int", "nullable": False, "required": True},
        }
    )

    class FakeVectorStores:
        def retrieve(self, *, vector_store_id):
            return SimpleNamespace(
                id=vector_store_id,
                metadata={"raghilda_attributes_schema_json": schema_json},
            )

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    store = OpenAIStore.connect(store_id="vs_test")
    assert store.attributes_schema == {"tenant": str, "priority": int}


def test_openai_store_connect_rejects_internal_attribute_names_from_metadata(
    monkeypatch,
):
    schema_json = json.dumps(
        {
            "_raghilda_origin": {"type": "str", "nullable": False, "required": True},
        }
    )

    class FakeVectorStores:
        def retrieve(self, *, vector_store_id):
            return SimpleNamespace(
                id=vector_store_id,
                metadata={"raghilda_attributes_schema_json": schema_json},
            )

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    with pytest.raises(ValueError, match="_raghilda_origin"):
        OpenAIStore.connect(store_id="vs_test")


def test_openai_store_connect_requires_attributes_schema_metadata(monkeypatch):
    class FakeVectorStores:
        def retrieve(self, *, vector_store_id):
            return SimpleNamespace(id=vector_store_id, metadata={})

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    with pytest.raises(ValueError, match="missing required key"):
        OpenAIStore.connect(store_id="vs_test")


def test_openai_store_connect_rejects_attributes_argument(monkeypatch):
    class FakeVectorStores:
        def retrieve(self, *, vector_store_id):
            return SimpleNamespace(id=vector_store_id, metadata={})

    fake_client = SimpleNamespace(vector_stores=FakeVectorStores())

    def fake_openai_client(*, api_key=None, base_url=None):
        return fake_client

    monkeypatch.setattr("raghilda._openai_store.openai.Client", fake_openai_client)

    with pytest.raises(TypeError, match="unexpected keyword argument 'attributes'"):
        cast(Any, OpenAIStore.connect)(
            store_id="vs_test",
            attributes={"tenant": str},
        )


def test_openai_store_insert_updates_when_snapshot_download_forbidden():
    old_content = "hello world"
    new_content = "hello world updated"
    old_hash = hashlib.sha256(old_content.encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": old_hash,
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            request = httpx.Request(
                "GET", f"https://api.openai.com/v1/files/{file_id}/content"
            )
            response = httpx.Response(400, request=request)
            raise openai.BadRequestError(
                "Not allowed to download files of purpose: assistants",
                response=response,
                body=None,
            )

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=new_content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "replaced"
    assert result.replaced_document is None
    assert fake_vector_store_files.deleted_ids == ["file_old"]
    assert len(fake_vector_store_files.upload_calls) == 1


def test_openai_store_insert_updates_when_snapshot_download_connection_error():
    new_content = "hello world updated"
    old_hash = hashlib.sha256("hello world".encode("utf-8")).hexdigest()

    class FakeVectorStoreFiles:
        def __init__(self):
            self.deleted_ids = []
            self.upload_calls = []
            self.page = _SinglePage(
                [
                    SimpleNamespace(
                        id="file_old",
                        created_at=1,
                        filename="doc.md",
                        attributes={
                            "_raghilda_origin": "doc",
                            "_raghilda_content_hash": old_hash,
                            "tenant": "old",
                        },
                    )
                ]
            )

        def list(self, **kwargs):
            return self.page

        def delete(self, file_id, **kwargs):
            self.deleted_ids.append(file_id)
            return SimpleNamespace(id=file_id, deleted=True)

        def upload_and_poll(self, **kwargs):
            self.upload_calls.append(kwargs)
            return SimpleNamespace(id="file_new")

    class FakeFiles:
        def content(self, file_id):
            request = httpx.Request(
                "GET", f"https://api.openai.com/v1/files/{file_id}/content"
            )
            raise openai.APIConnectionError(request=request)

    fake_vector_store_files = FakeVectorStoreFiles()
    fake_client = SimpleNamespace(
        vector_stores=SimpleNamespace(files=fake_vector_store_files),
        files=FakeFiles(),
    )
    store = OpenAIStore(
        client=fake_client,
        store_id="vs_test",
        attributes={"tenant": str},
    )

    result = store.upsert(
        MarkdownDocument(
            origin="doc",
            content=new_content,
            attributes={"tenant": "new"},
        )
    )

    assert result.action == "replaced"
    assert result.replaced_document is None
    assert fake_vector_store_files.deleted_ids == ["file_old"]
    assert len(fake_vector_store_files.upload_calls) == 1


def _get_markdown_chunk(doc, start, end):
    return MarkdownChunk(
        start_index=start,
        end_index=end,
        text=doc.content[start:end],
        token_count=len(doc.content[start:end]),
    )


def test_ingest():
    _skip_if_unset("OPENAI_API_KEY")
    from raghilda.chunker import MarkdownChunker
    from raghilda.read import read_as_markdown

    links = find_links("https://r4ds.hadley.nz/base-R.html", validate=True)
    links = links[:3]

    store = DuckDBStore.create(
        location=":memory:",
        embed=EmbeddingOpenAI(),
        overwrite=True,
        name="ingest_db",
        title="Ingest Test DuckDB Store",
    )

    # Use smaller chunks to avoid exceeding embedding token limits on code-heavy pages
    chunker = MarkdownChunker(chunk_size=800)

    def prepare(uri: str):
        return chunker.chunk_document(read_as_markdown(uri))

    store.ingest(links, prepare=prepare)


def test_ingest_with_generator():
    """Test that ingest works with a generator (iterable without __len__)."""
    from raghilda.chunker import MarkdownChunker

    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_generator_db",
    )

    chunker = MarkdownChunker()

    def make_docs():
        for i in range(3):
            doc = MarkdownDocument(origin=f"test_{i}", content=f"Document number {i}")
            yield doc

    # Use identity prepare since we're yielding MarkdownDocuments that need chunking
    store.ingest(
        make_docs(), prepare=lambda doc: chunker.chunk_document(doc), progress=False
    )
    assert store.size() == 3


def test_ingest_with_custom_prepare():
    """Test that ingest works with a custom prepare function."""
    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_prepare_db",
    )

    def custom_prepare(item: dict) -> MarkdownDocument:
        doc = MarkdownDocument(origin=item["id"], content=item["text"])
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(item["text"]),
                text=item["text"],
                token_count=len(item["text"]),
            )
        ]
        return doc

    items = [
        {"id": "doc1", "text": "First document content"},
        {"id": "doc2", "text": "Second document content"},
    ]

    store.ingest(items, prepare=custom_prepare, progress=False)
    assert store.size() == 2


def test_ingest_lazy_evaluation():
    """Test that ingest consumes the iterator lazily, not all at once."""
    import threading
    import time

    store = DuckDBStore.create(
        location=":memory:",
        embed=None,
        overwrite=True,
        name="ingest_lazy_db",
    )

    consumed_count = 0
    max_concurrent_consumed = 0
    inserted_count = 0
    lock = threading.Lock()

    def tracking_generator():
        nonlocal consumed_count, max_concurrent_consumed
        for i in range(20):
            with lock:
                consumed_count += 1
                # Track max items consumed while inserts are still pending
                pending = consumed_count - inserted_count
                if pending > max_concurrent_consumed:
                    max_concurrent_consumed = pending
            yield {"id": f"doc_{i}", "text": f"Document content {i}"}

    def slow_prepare(item: dict) -> MarkdownDocument:
        nonlocal inserted_count
        # Simulate slow processing
        time.sleep(0.05)
        doc = MarkdownDocument(origin=item["id"], content=item["text"])
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=len(item["text"]),
                text=item["text"],
                token_count=len(item["text"]),
            )
        ]
        with lock:
            inserted_count += 1
        return doc

    # Use only 2 workers to make the test more deterministic
    store.ingest(
        tracking_generator(), prepare=slow_prepare, num_workers=2, progress=False
    )

    assert store.size() == 20
    # With lazy evaluation, max concurrent should be bounded by num_workers (plus some buffer)
    # If it consumed eagerly, all 20 would be consumed before any inserts complete
    assert max_concurrent_consumed <= 10, (
        f"Expected lazy consumption but {max_concurrent_consumed} items were consumed "
        f"concurrently, suggesting eager evaluation"
    )


def test_connect(tmp_path):
    _skip_if_unset("OPENAI_API_KEY")
    db_path = tmp_path / "test.db"

    # Create a store with embeddings
    store = DuckDBStore.create(
        location=str(db_path),
        embed=EmbeddingOpenAI(model="text-embedding-3-small"),
        name="connect_test",
        title="Connect Test Store",
    )
    doc = MarkdownDocument(origin="test", content="hello world")
    doc.chunks = [_get_markdown_chunk(doc, start=0, end=5)]
    store.upsert(doc)
    store.build_index()
    store.con.close()

    # Connect to existing store
    store2 = DuckDBStore.connect(str(db_path))
    assert store2.metadata.name == "connect_test"
    assert store2.metadata.title == "Connect Test Store"
    assert store2.size() == 1

    # Embedding provider should be restored
    assert store2.metadata.embed is not None
    assert isinstance(store2.metadata.embed, EmbeddingOpenAI)
    assert store2.metadata.embed.model == "text-embedding-3-small"

    # Retrieve should work (uses both BM25 and VSS)
    results = store2.retrieve("hello", top_k=1)
    assert len(results) >= 1
    assert results[0].text == "hello"


def test_connect_fails_fast_when_embeddings_table_lacks_chunk_text(tmp_path):
    db_path = tmp_path / "missing_chunk_text.db"
    con = duckdb.connect(str(db_path))
    con.execute(
        """
        CREATE TABLE metadata (
            name VARCHAR,
            title VARCHAR,
            embed_config VARCHAR,
            attributes_schema_json VARCHAR
        )
        """
    )
    con.execute(
        "INSERT INTO metadata VALUES (?, ?, ?, ?)",
        ["test", "Test", None, json.dumps({})],
    )
    con.execute(
        """
        CREATE TABLE documents (
            doc_id VARCHAR,
            origin VARCHAR,
            text VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE embeddings (
            doc_id VARCHAR,
            chunk_id INTEGER,
            start_index INTEGER,
            end_index INTEGER,
            context VARCHAR
        )
        """
    )
    con.close()

    with pytest.raises(
        ValueError,
        match="table 'embeddings' missing required columns: chunk_text",
    ):
        DuckDBStore.connect(str(db_path))


def test_duckdb_store_does_not_require_pandas():
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    script = textwrap.dedent(
        """
        import builtins

        original_import = builtins.__import__

        def blocked_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pandas" or name.startswith("pandas."):
                raise ModuleNotFoundError("No module named 'pandas'")
            return original_import(name, globals, locals, fromlist, level)

        builtins.__import__ = blocked_import

        from raghilda.chunk import MarkdownChunk
        from raghilda.document import MarkdownDocument
        from raghilda.store import DuckDBStore

        store = DuckDBStore.create(location=":memory:", embed=None, overwrite=True)
        doc = MarkdownDocument(origin="test", content="hello world")
        doc.chunks = [
            MarkdownChunk(
                start_index=0,
                end_index=5,
                text="hello",
                token_count=5,
            )
        ]
        store.upsert(doc)
        assert store.size() == 1
        """
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = (
        f"{src_path}{os.pathsep}{env['PYTHONPATH']}"
        if env.get("PYTHONPATH")
        else str(src_path)
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        cwd=repo_root,
        env=env,
        text=True,
    )
    assert result.returncode == 0, result.stderr
