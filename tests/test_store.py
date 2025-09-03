from ragnar.store import DuckDBStore


class TestDuckDBStore:
    def test_create_duckdb_store(self):
        store = DuckDBStore.create(
            location=":memory:",
            embed=None,
            overwrite=True,
            name="test_db",
            title="Test DuckDB Store",
        )
        assert isinstance(store, DuckDBStore)
        assert store.metadata.name == "test_db"
        assert store.metadata.title == "Test DuckDB Store"
        assert store.metadata.embed is None
