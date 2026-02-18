import os
import socket
from typing import Annotated
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


def _can_reach_openai(timeout: float = 2.0) -> bool:
    try:
        with socket.create_connection(("api.openai.com", 443), timeout=timeout):
            return True
    except OSError:
        return False


def _require_openai_integration() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set in environment variables")
    if not _can_reach_openai():
        pytest.skip("OpenAI API is not reachable from this environment")


class TestDuckDBStore:
    @pytest.fixture
    def embed(self, request):
        try:
            value = request.param
            if isinstance(value, EmbeddingOpenAI):
                _require_openai_integration()
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
        store.insert(doc)
        return store

    def test_create_store(self, store):
        assert isinstance(store, DuckDBStore)
        assert store.metadata.name == "test_db"
        assert store.metadata.title == "Test DuckDB Store"
        assert store.metadata.embed is None

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_insert(self, store_with_docs):
        assert store_with_docs.size() == 1

    @pytest.mark.parametrize("embed", [EmbeddingOpenAI()], indirect=True)
    def test_retrieve_vss(self, store_with_docs):
        results = store_with_docs.retrieve_vss("test", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

        results = store_with_docs.retrieve_vss("test", top_k=5)
        assert len(results) == 5

    @pytest.mark.parametrize("embed", [None, EmbeddingOpenAI()], indirect=True)
    def test_retrieve_bm25(self, store_with_docs):
        store_with_docs.build_index("bm25")
        results = store_with_docs.retrieve_bm25("document", top_k=3)
        assert len(results) == 3
        for chunk in results:
            assert isinstance(chunk, RetrievedDuckDBMarkdownChunk)
            assert chunk.text is not None

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
        store.insert(doc)
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
        store.insert(doc)
        store.build_index("bm25")

        results = store.retrieve("beta", top_k=5, deoverlap=True)

        assert len(results) == 1
        assert results[0].context == "h1"
        assert results[0].attributes == {"topic": ["first", "second"]}

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
        store.insert(doc)
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

        store.insert(doc)
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

        store.insert(doc)
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

        store.insert(doc)
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

        store.insert(doc)
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

        store.insert(doc)
        store.build_index("bm25")

        results = store.retrieve("alpha", top_k=5, deoverlap=False)
        assert len(results) == 1
        assert results[0].attributes == {
            "tenant": "docs",
            "priority": 0,
            "is_public": False,
            "topic": None,
        }

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
            store.insert(doc)

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
            store.insert(doc)

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
            store.insert(doc)

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
            store.insert(doc)

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
            store.insert(doc)

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
        _require_openai_integration()

    @pytest.fixture
    def store(self):
        store = OpenAIStore.create()
        try:
            yield store
        finally:
            store.client.vector_stores.delete(vector_store_id=store.store_id)

    @pytest.fixture
    def store_with_attributes(self):
        store = OpenAIStore.create(attributes={"tenant": str, "priority": int})
        try:
            yield store
        finally:
            store.client.vector_stores.delete(vector_store_id=store.store_id)

    @pytest.fixture
    def store_with_class_attributes(self):
        class AttributesSpec:
            tenant: str
            priority: int

        store = OpenAIStore.create(attributes=AttributesSpec)
        try:
            yield store
        finally:
            store.client.vector_stores.delete(vector_store_id=store.store_id)

    @pytest.fixture
    def store_with_docs(self, store):
        doc = MarkdownDocument(
            origin="test", content="hello world this is a document world world world"
        )
        store.insert(doc)
        return store

    def test_create_store(self, store):
        assert isinstance(store, OpenAIStore)
        assert isinstance(store.store_id, str)

    def test_insert(self, store_with_docs):
        assert store_with_docs.size() == 1

    def test_retrieve(self, store_with_docs):
        for _ in range(3):
            store_with_docs.insert(
                MarkdownDocument(
                    origin="test", content="hello world world world world world"
                )
            )
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
        store_with_attributes.insert(
            MarkdownDocument(
                origin="doc-attrs",
                content="alpha bronze owl",
                attributes={"tenant": "docs", "priority": 2},
            )
        )
        results = store_with_attributes.retrieve(
            "bronze owl",
            top_k=5,
            attributes_filter="tenant = 'docs' AND priority = 2",
        )
        assert len(results) > 0
        assert all(chunk.attributes["tenant"] == "docs" for chunk in results)
        assert all(float(chunk.attributes["priority"]) == 2.0 for chunk in results)

    def test_retrieve_supports_attributes_filter(self, store_with_attributes):
        store_with_attributes.insert(
            MarkdownDocument(
                origin="docs-priority-2",
                content="alpha alpha alpha",
                attributes={"tenant": "docs", "priority": 2},
            )
        )
        store_with_attributes.insert(
            MarkdownDocument(
                origin="docs-priority-1",
                content="alpha beta",
                attributes={"tenant": "docs", "priority": 1},
            )
        )
        store_with_attributes.insert(
            MarkdownDocument(
                origin="ops-priority-5",
                content="alpha gamma",
                attributes={"tenant": "ops", "priority": 5},
            )
        )

        results = store_with_attributes.retrieve(
            "alpha",
            top_k=10,
            attributes_filter="tenant = 'docs' AND priority >= 2",
        )

        assert len(results) > 0
        assert all(chunk.attributes["tenant"] == "docs" for chunk in results)
        assert all(float(chunk.attributes["priority"]) >= 2.0 for chunk in results)

    def test_retrieve_supports_attributes_filter_ast(self, store_with_attributes):
        store_with_attributes.insert(
            MarkdownDocument(
                origin="docs-priority-3",
                content="alpha alpha delta",
                attributes={"tenant": "docs", "priority": 3},
            )
        )
        store_with_attributes.insert(
            MarkdownDocument(
                origin="ops-priority-3",
                content="alpha alpha epsilon",
                attributes={"tenant": "ops", "priority": 3},
            )
        )
        results = store_with_attributes.retrieve(
            "alpha",
            top_k=10,
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

    def test_rejects_chunk_attributes(self, store_with_attributes):
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
            ValueError, match="OpenAIStore does not support per-chunk attributes"
        ):
            store_with_attributes.insert(doc)


def test_openai_store_create_rejects_vector_attributes_schema():
    with pytest.raises(ValueError, match="Vector attribute types are not supported"):
        OpenAIStore.create(attributes={"embedding25": Annotated[list[float], 25]})


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


def _get_markdown_chunk(doc, start, end):
    return MarkdownChunk(
        start_index=start,
        end_index=end,
        text=doc.content[start:end],
        token_count=len(doc.content[start:end]),
    )


def test_ingest():
    _require_openai_integration()
    from raghilda.chunker import MarkdownChunker
    from raghilda.read import read_as_markdown

    links = find_links("https://r4ds.hadley.nz/base-R.html", validate=True)

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
    _require_openai_integration()
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
    store.insert(doc)
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
