import pytest
import psycopg2
from raghilda._postgres_store import PostgreSQLStore
from raghilda._embedding import EmbeddingProvider, EmbedInputType
from raghilda.embedding import register_embedding_provider
from raghilda.document import MarkdownDocument
from raghilda.chunker import MarkdownChunker


POSTGRES_URL = "postgresql://raghilda:raghilda@localhost:5432/raghilda"


@register_embedding_provider("FakeEmbedding")
class FakeEmbedding(EmbeddingProvider):
    def embed(self, x, input_type: EmbedInputType = EmbedInputType.DOCUMENT):
        return [[1.0, 2.0, 3.0]] * len(x)

    def get_config(self):
        return {"type": "FakeEmbedding"}

    @classmethod
    def from_config(cls, config):
        return cls()


@pytest.fixture
def pg_con():
    try:
        con = psycopg2.connect(POSTGRES_URL)
    except psycopg2.OperationalError:
        pytest.skip("PostgreSQL not available at localhost:5432")
    con.autocommit = True
    with con.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS embeddings;")
        cur.execute("DROP TABLE IF EXISTS documents;")
        cur.execute("DROP TABLE IF EXISTS metadata;")
    con.autocommit = False
    yield con
    con.close()


def test_create_store(pg_con):
    store = PostgreSQLStore.create(
        con=pg_con,
        embed=FakeEmbedding(),
        name="test_store",
        title="Test Store",
    )
    assert store._metadata["name"] == "test_store"
    assert store._metadata["title"] == "Test Store"

    with pg_con.cursor() as cur:
        cur.execute("SELECT name, title, embed_config FROM metadata;")
        row = cur.fetchone()
        assert row[0] == "test_store"
        assert row[1] == "Test Store"
        assert '"type": "FakeEmbedding"' in row[2]

        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'embeddings'
            ORDER BY ordinal_position;
        """)
        columns = {r[0]: r[1] for r in cur.fetchall()}
        assert "origin" in columns
        assert "chunk_id" in columns
        assert "start_index" in columns
        assert "end_index" in columns
        assert "char_count" in columns
        assert "context" in columns
        assert "embedding" in columns


def test_create_store_with_attributes(pg_con):
    store = PostgreSQLStore.create(
        con=pg_con,
        embed=FakeEmbedding(),
        attributes={"tenant": str, "priority": int, "score": float, "active": bool},
    )
    assert "tenant" in store._metadata["attributes"]
    assert "priority" in store._metadata["attributes"]

    with pg_con.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'embeddings'
            ORDER BY ordinal_position;
        """)
        columns = {r[0]: r[1] for r in cur.fetchall()}
        assert columns["tenant"] == "character varying"
        assert columns["priority"] == "integer"
        assert columns["score"] == "double precision"
        assert columns["active"] == "boolean"
        assert "embedding" in columns


def test_create_store_with_struct_attribute(pg_con):
    store = PostgreSQLStore.create(
        con=pg_con,
        embed=FakeEmbedding(),
        attributes={"meta": {"key": str, "value": int}},
    )
    assert "meta" in store._metadata["attributes"]

    with pg_con.cursor() as cur:
        cur.execute("""
            SELECT column_name, data_type
            FROM information_schema.columns
            WHERE table_name = 'embeddings' AND column_name = 'meta';
        """)
        row = cur.fetchone()
        assert row is not None
        assert row[1] == "jsonb"


def test_create_store_no_embed(pg_con):
    store = PostgreSQLStore.create(
        con=pg_con,
        embed=None,
    )
    assert store._metadata["name"] == "raghilda_db"

    with pg_con.cursor() as cur:
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_name = 'embeddings';
        """)
        columns = {r[0] for r in cur.fetchall()}
        assert "embedding" not in columns


def test_connect_recovers_metadata(pg_con):
    PostgreSQLStore.create(
        con=pg_con,
        embed=FakeEmbedding(),
        name="my_store",
        title="My Store",
        attributes={"tenant": str, "priority": int},
    )

    store = PostgreSQLStore.connect(pg_con)
    assert store._metadata["name"] == "my_store"
    assert store._metadata["title"] == "My Store"
    assert isinstance(store._metadata["embed"], FakeEmbedding)
    assert "tenant" in store._metadata["attributes"]
    assert "priority" in store._metadata["attributes"]


def test_connect_no_embed(pg_con):
    PostgreSQLStore.create(
        con=pg_con,
        embed=None,
        name="no_embed_store",
    )

    store = PostgreSQLStore.connect(pg_con)
    assert store._metadata["name"] == "no_embed_store"
    assert store._metadata["embed"] is None


def test_connect_no_metadata(pg_con):
    with pytest.raises(ValueError, match="No metadata found"):
        PostgreSQLStore.connect(pg_con)


def _make_chunked_doc(origin="doc1", content="# Hello\n\nThis is a test document."):
    doc = MarkdownDocument(content=content, origin=origin)
    chunker = MarkdownChunker(chunk_size=500)
    return chunker.chunk(doc)


def test_upsert_insert(pg_con):
    store = PostgreSQLStore.create(con=pg_con, embed=FakeEmbedding())
    doc = _make_chunked_doc()
    result = store.upsert(doc)
    assert result.action == "inserted"

    with pg_con.cursor() as cur:
        cur.execute("SELECT origin, text FROM documents;")
        row = cur.fetchone()
        assert row[0] == "doc1"
        assert row[1] == doc.content

        cur.execute("SELECT origin, start_index, end_index FROM embeddings;")
        rows = cur.fetchall()
        assert len(rows) == len(doc.chunks)


def test_upsert_skip_unchanged(pg_con):
    store = PostgreSQLStore.create(con=pg_con, embed=FakeEmbedding())
    doc = _make_chunked_doc()
    store.upsert(doc)

    result = store.upsert(doc)
    assert result.action == "skipped"


def test_upsert_replace(pg_con):
    store = PostgreSQLStore.create(con=pg_con, embed=FakeEmbedding())
    doc1 = _make_chunked_doc(content="# Hello\n\nOriginal content.")
    store.upsert(doc1)

    doc2 = _make_chunked_doc(content="# Hello\n\nUpdated content.")
    result = store.upsert(doc2)
    assert result.action == "replaced"

    with pg_con.cursor() as cur:
        cur.execute("SELECT text FROM documents WHERE origin = %s;", ["doc1"])
        row = cur.fetchone()
        assert row[0] == doc2.content
