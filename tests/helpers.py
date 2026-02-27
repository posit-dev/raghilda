import os
import socket
from functools import lru_cache

import pytest


@lru_cache(maxsize=None)
def _can_reach_network(host: str, timeout: float) -> bool:
    try:
        with socket.create_connection((host, 443), timeout=timeout):
            return True
    except OSError:
        return False


def _skip_if_unavailable(
    *,
    env_vars: tuple[str, ...],
    host: str,
    timeout: float = 2.0,
) -> None:
    if not any(os.environ.get(var) for var in env_vars):
        joined_env_vars = " or ".join(env_vars)
        pytest.skip(f"{joined_env_vars} not set in environment variables")

    if not _can_reach_network(host, timeout):
        pytest.skip(f"Network is not reachable ({host})")


def skip_if_no_openai() -> None:
    _skip_if_unavailable(env_vars=("OPENAI_API_KEY",), host="api.openai.com")


def skip_if_no_cohere() -> None:
    _skip_if_unavailable(
        env_vars=("CO_API_KEY", "COHERE_API_KEY", "CHROMA_COHERE_API_KEY"),
        host="api.cohere.com",
    )
