import pytest
import psycopg2
from raghilda._postgres_store import PostgreSQLStore
from raghilda._embedding import EmbeddingProvider, EmbedInputType


POSTGRES_URL = "postgresql://raghilda:raghilda@localhost:5432/raghilda"


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
