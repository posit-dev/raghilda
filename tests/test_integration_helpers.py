import pytest
from tests import helpers as test_helpers


def test_skip_if_no_openai_skips_when_env_var_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(pytest.skip.Exception):
        test_helpers.skip_if_no_openai()


def test_skip_if_no_openai_skips_when_network_unreachable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    test_helpers._can_reach_network.cache_clear()

    def fail_create_connection(*_args, **_kwargs):
        raise OSError()

    monkeypatch.setattr(
        test_helpers.socket, "create_connection", fail_create_connection
    )

    with pytest.raises(pytest.skip.Exception):
        test_helpers.skip_if_no_openai()

    test_helpers._can_reach_network.cache_clear()


def test_skip_if_no_openai_passes_when_env_and_network_are_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    test_helpers._can_reach_network.cache_clear()

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        test_helpers.socket,
        "create_connection",
        lambda *_args, **_kwargs: _Connection(),
    )

    test_helpers.skip_if_no_openai()
    test_helpers._can_reach_network.cache_clear()


def test_skip_if_no_openai_caches_network_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    test_helpers._can_reach_network.cache_clear()

    calls = 0

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def fake_create_connection(_address, timeout):
        nonlocal calls
        assert timeout == 2.0
        calls += 1
        return _Connection()

    monkeypatch.setattr(
        test_helpers.socket, "create_connection", fake_create_connection
    )

    test_helpers.skip_if_no_openai()
    test_helpers.skip_if_no_openai()

    assert calls == 1
    test_helpers._can_reach_network.cache_clear()


def test_skip_if_no_cohere_skips_with_only_chroma_cohere_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CHROMA_COHERE_API_KEY", "test-key")

    with pytest.raises(pytest.skip.Exception):
        test_helpers.skip_if_no_cohere()


def test_skip_if_no_cohere_chroma_accepts_chroma_cohere_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("CO_API_KEY", raising=False)
    monkeypatch.delenv("COHERE_API_KEY", raising=False)
    monkeypatch.setenv("CHROMA_COHERE_API_KEY", "test-key")
    test_helpers._can_reach_network.cache_clear()

    class _Connection:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        test_helpers.socket,
        "create_connection",
        lambda *_args, **_kwargs: _Connection(),
    )
    monkeypatch.setattr(
        test_helpers.pytest,
        "skip",
        lambda message: (_ for _ in ()).throw(
            AssertionError(f"unexpected skip: {message}")
        ),
    )

    test_helpers.skip_if_no_cohere_chroma()
    test_helpers._can_reach_network.cache_clear()
