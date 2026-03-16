# These tests require a running PostgreSQL instance with pgvector.
# Start one locally with:
#
#   docker run -d --name raghilda-postgres \
#     -e POSTGRES_USER=raghilda \
#     -e POSTGRES_PASSWORD=raghilda \
#     -e POSTGRES_DB=raghilda_test \
#     -p 5432:5432 pgvector/pgvector:pg17
#
# Tests are automatically skipped when PostgreSQL is not available.

import pytest
from tests import helpers as test_helpers
from raghilda.store import PostgresStore
from raghilda.document import MarkdownDocument
from raghilda.chunk import MarkdownChunk, RetrievedChunk
from raghilda._embedding import EmbeddingProvider, EmbedInputType


_BASE_CONN = "postgresql://raghilda:raghilda@localhost:5432"


class CountingEmbedding(EmbeddingProvider):
    def __init__(self):
        self.calls = 0

    def embed(self, x, input_type: EmbedInputType = EmbedInputType.DOCUMENT):
        self.calls += 1
        return [[float(len(text))] for text in x]

    def get_config(self):
        return {"type": "CountingEmbedding"}

    @classmethod
    def from_config(cls, config):
        return cls()


def _chunk(text, start, end, context=None, attributes=None):
    return MarkdownChunk(
        text=text,
        start_index=start,
        end_index=end,
        char_count=len(text),
        context=context,
        attributes=attributes,
    )


def _chunk_from_doc(doc, start, end, context=None):
    return _chunk(doc.content[start:end], start, end, context=context)


def _conn_str(dbname: str) -> str:
    return f"{_BASE_CONN}/{dbname}"


def _drop_db(dbname: str) -> None:
    import psycopg

    with psycopg.connect(f"{_BASE_CONN}/raghilda_test", autocommit=True) as con:
        con.execute(f'DROP DATABASE IF EXISTS "{dbname}"')


class TestPostgresStore:
    @pytest.fixture
    def embed(self, request):
        value = getattr(request, "param", None)
        if value == "openai":
            test_helpers.skip_if_no_openai()
            from raghilda.embedding import EmbeddingOpenAI

            return EmbeddingOpenAI()
        return value

    @pytest.fixture
    def store(self, embed, request):
        test_helpers.skip_if_no_postgres()
        dbname = f"raghilda_test_{request.node.name[:40]}"
        _drop_db(dbname)
        store = PostgresStore.create(
            connection_string=_conn_str(dbname),
            embed=embed,
            overwrite=True,
            name="test_db",
            title="Test PostgreSQL Store",
        )
        yield store
        store.con.close()
        _drop_db(dbname)

    @pytest.fixture
    def store_with_docs(self, store):
        doc = MarkdownDocument(origin="test", content="This is a test document.")
        from raghilda.document import ChunkedMarkdownDocument

        chunked = ChunkedMarkdownDocument(
            origin=doc.origin,
            content=doc.content,
            chunks=[
                _chunk_from_doc(doc, 0, 5, context="test"),
                _chunk_from_doc(doc, 5, 10, context="test"),
                _chunk_from_doc(doc, 10, 15, context="test"),
                _chunk_from_doc(doc, 15, 24, context="test"),
                _chunk_from_doc(doc, 8, 24, context="test"),
            ],
        )
        store.upsert(chunked)
        return store

    def test_create_store(self, store):
        assert store.metadata.name == "test_db"
        assert store.metadata.title == "Test PostgreSQL Store"

    def test_insert(self, store):
        doc = MarkdownDocument(origin="doc1", content="Hello world")
        chunks = [
            _chunk("Hello world", 0, 11),
        ]
        from raghilda.document import ChunkedMarkdownDocument

        chunked = ChunkedMarkdownDocument(
            origin=doc.origin, content=doc.content, chunks=chunks
        )
        result = store.upsert(chunked)
        assert result.action == "inserted"
        assert store.size() == 1

    def test_insert_same_origin_skips_unchanged_by_default(self, store):
        embed = CountingEmbedding()
        test_helpers.skip_if_no_postgres()
        dbname = "raghilda_test_skip_unchanged"
        _drop_db(dbname)
        s = PostgresStore.create(
            connection_string=_conn_str(dbname),
            embed=embed,
            overwrite=True,
        )
        calls_after_create = embed.calls  # create() probes embedding size
        doc = MarkdownDocument(origin="doc1", content="Hello world")
        chunks = [
            _chunk("Hello world", 0, 11),
        ]
        from raghilda.document import ChunkedMarkdownDocument

        chunked = ChunkedMarkdownDocument(
            origin=doc.origin, content=doc.content, chunks=chunks
        )
        r1 = s.upsert(chunked)
        assert r1.action == "inserted"
        assert embed.calls == calls_after_create + 1

        r2 = s.upsert(chunked)
        assert r2.action == "skipped"
        assert embed.calls == calls_after_create + 1  # no extra embed call

        s.con.close()
        _drop_db(dbname)

    def test_insert_same_origin_replaces_when_content_changes(self, store):
        embed = CountingEmbedding()
        test_helpers.skip_if_no_postgres()
        dbname = "raghilda_test_replace"
        _drop_db(dbname)
        s = PostgresStore.create(
            connection_string=_conn_str(dbname),
            embed=embed,
            overwrite=True,
        )
        from raghilda.document import ChunkedMarkdownDocument

        doc1 = ChunkedMarkdownDocument(
            origin="doc1",
            content="Hello world",
            chunks=[
                _chunk("Hello world", 0, 11),
            ],
        )
        r1 = s.upsert(doc1)
        assert r1.action == "inserted"

        doc2 = ChunkedMarkdownDocument(
            origin="doc1",
            content="Goodbye world",
            chunks=[
                _chunk("Goodbye world", 0, 13),
            ],
        )
        r2 = s.upsert(doc2)
        assert r2.action == "replaced"
        assert r2.replaced_document is not None
        assert r2.replaced_document.content == "Hello world"
        assert s.size() == 1

        s.con.close()
        _drop_db(dbname)

    @pytest.mark.parametrize("embed", ["openai"], indirect=True)
    def test_retrieve_vss(self, store_with_docs):
        chunks = store_with_docs.retrieve_vss("test document", top_k=3)
        assert len(chunks) > 0
        assert all(isinstance(c, RetrievedChunk) for c in chunks)
        assert all(len(c.metrics) > 0 for c in chunks)

    def test_retrieve_vss_returns_document_slice_for_non_zero_start(
        self, store_with_docs
    ):
        """Guard against 0/1-indexed slicing bugs."""
        embed = CountingEmbedding()
        test_helpers.skip_if_no_postgres()
        dbname = "raghilda_test_vss_slice"
        _drop_db(dbname)
        s = PostgresStore.create(
            connection_string=_conn_str(dbname),
            embed=embed,
            overwrite=True,
        )
        from raghilda.document import ChunkedMarkdownDocument

        doc = ChunkedMarkdownDocument(
            origin="doc1",
            content="aaabbbccc",
            chunks=[
                _chunk("bbb", 3, 6),
            ],
        )
        s.upsert(doc)
        chunks = s.retrieve_vss([1.0], top_k=1)
        assert len(chunks) == 1
        assert chunks[0].text == "bbb"
        assert chunks[0].start_index == 3
        assert chunks[0].end_index == 6

        s.con.close()
        _drop_db(dbname)

    def test_retrieve_fts(self, store_with_docs):
        chunks = store_with_docs.retrieve_fts("test document", top_k=3)
        assert len(chunks) > 0
        assert all(isinstance(c, RetrievedChunk) for c in chunks)

    def test_size(self, store):
        assert store.size() == 0
        from raghilda.document import ChunkedMarkdownDocument

        doc = ChunkedMarkdownDocument(
            origin="doc1",
            content="Hello",
            chunks=[_chunk("Hello", 0, 5)],
        )
        store.upsert(doc)
        assert store.size() == 1

    def test_create_store_with_attributes_schema(self):
        test_helpers.skip_if_no_postgres()
        dbname = "raghilda_test_attrs"
        _drop_db(dbname)
        s = PostgresStore.create(
            connection_string=_conn_str(dbname),
            embed=CountingEmbedding(),
            overwrite=True,
            attributes={"tenant": str, "priority": int, "is_public": bool},
        )
        from raghilda.document import ChunkedMarkdownDocument

        doc = ChunkedMarkdownDocument(
            origin="doc1",
            content="Hello",
            chunks=[
                _chunk(
                    "Hello",
                    0,
                    5,
                    attributes={
                        "tenant": "docs",
                        "priority": 1,
                        "is_public": True,
                    },
                )
            ],
        )
        result = s.upsert(doc)
        assert result.action == "inserted"

        chunks = s.retrieve_vss([1.0], top_k=1)
        assert len(chunks) == 1
        assert chunks[0].attributes["tenant"] == "docs"
        assert chunks[0].attributes["priority"] == 1
        assert chunks[0].attributes["is_public"] is True

        s.con.close()
        _drop_db(dbname)

    def test_insert_and_retrieve_with_attributes_filter(self):
        test_helpers.skip_if_no_postgres()
        dbname = "raghilda_test_filter"
        _drop_db(dbname)
        s = PostgresStore.create(
            connection_string=_conn_str(dbname),
            embed=CountingEmbedding(),
            overwrite=True,
            attributes={"tenant": str},
        )
        from raghilda.document import ChunkedMarkdownDocument

        doc1 = ChunkedMarkdownDocument(
            origin="doc1",
            content="Hello",
            chunks=[_chunk("Hello", 0, 5, attributes={"tenant": "a"})],
        )
        doc2 = ChunkedMarkdownDocument(
            origin="doc2",
            content="World",
            chunks=[_chunk("World", 0, 5, attributes={"tenant": "b"})],
        )
        s.upsert(doc1)
        s.upsert(doc2)

        chunks = s.retrieve_vss([1.0], top_k=10, attributes_filter="tenant = 'a'")
        assert len(chunks) == 1
        assert chunks[0].origin == "doc1"

        s.con.close()
        _drop_db(dbname)
