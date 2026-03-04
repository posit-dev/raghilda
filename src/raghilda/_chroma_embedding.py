from __future__ import annotations

from functools import singledispatch
import os
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Optional,
    Sequence,
    TypeAlias,
    TypeVar,
    cast,
)

from ._embedding import (
    EmbeddingCohere,
    EmbeddingOpenAI,
    EmbeddingProvider,
    EmbedInputType,
    embedding_from_config,
)

if TYPE_CHECKING:
    import numpy as np
    from chromadb.api.types import Documents, EmbeddingFunction

    ChromaEmbeddingFunction: TypeAlias = EmbeddingFunction[Documents]
    ChromaEmbedding: TypeAlias = EmbeddingProvider | ChromaEmbeddingFunction
else:
    ChromaEmbeddingFunction = Any
    ChromaEmbedding = Any


_ADAPTER_NAME = "raghilda_embedding_adapter"
_EmbeddingFunctionBase: type[Any]


try:
    from chromadb import EmbeddingFunction as _ChromaEmbeddingFunctionBase
    from chromadb.utils.embedding_functions import register_embedding_function

    _EmbeddingFunctionBase = _ChromaEmbeddingFunctionBase
    _CHROMA_AVAILABLE = True
except ImportError:
    _CHROMA_AVAILABLE = False
    register_embedding_function = None

    class _FallbackEmbeddingFunctionBase:
        pass

    _EmbeddingFunctionBase = _FallbackEmbeddingFunctionBase


class ChromaEmbeddingAdapter(_EmbeddingFunctionBase):
    """Adapter to use any raghilda EmbeddingProvider with ChromaDB."""

    def __init__(self, provider: EmbeddingProvider) -> None:
        if not _CHROMA_AVAILABLE:
            raise ModuleNotFoundError(
                "ChromaDB is required to use ChromaDBStore. Install with `pip install chromadb`."
            )
        self._provider = provider

    def __call__(self, input: Sequence[str]) -> list[np.ndarray]:
        import numpy as np

        embeddings = self._provider.embed(list(input), EmbedInputType.DOCUMENT)
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    def embed_query(self, input: Sequence[str]) -> list[np.ndarray]:
        import numpy as np

        embeddings = self._provider.embed(list(input), EmbedInputType.QUERY)
        return [np.array(emb, dtype=np.float32) for emb in embeddings]

    @staticmethod
    def name() -> str:
        return _ADAPTER_NAME

    def get_config(self) -> dict[str, Any]:
        return {"provider_config": self._provider.get_config()}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "ChromaEmbeddingAdapter":
        provider_config = config.get("provider_config", {})
        provider = embedding_from_config(provider_config)
        return ChromaEmbeddingAdapter(provider)


if register_embedding_function is not None:
    register_embedding_function(ChromaEmbeddingAdapter)


P = TypeVar("P", bound=EmbeddingProvider)


@singledispatch
def to_chroma_embedding_function(
    provider: EmbeddingProvider,
) -> ChromaEmbeddingFunction:
    return cast(ChromaEmbeddingFunction, ChromaEmbeddingAdapter(provider))


def register_embedding_converter(
    provider_type: type[P],
) -> Callable[
    [Callable[[P], ChromaEmbeddingFunction]],
    Callable[[P], ChromaEmbeddingFunction],
]:
    assert issubclass(provider_type, EmbeddingProvider)

    def decorator(
        converter: Callable[[P], ChromaEmbeddingFunction],
    ) -> Callable[[P], ChromaEmbeddingFunction]:
        to_chroma_embedding_function.register(provider_type)(converter)
        return converter

    return decorator


@register_embedding_converter(EmbeddingOpenAI)
def _convert_openai_provider(provider: EmbeddingOpenAI) -> ChromaEmbeddingFunction:
    from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

    if os.getenv("CHROMA_OPENAI_API_KEY"):
        return OpenAIEmbeddingFunction(
            model_name=provider.model,
            api_base=provider.base_url,
        )
    if provider.api_key is None or provider.api_key == os.getenv("OPENAI_API_KEY"):
        return OpenAIEmbeddingFunction(
            model_name=provider.model,
            api_base=provider.base_url,
            api_key_env_var="OPENAI_API_KEY",
        )
    return OpenAIEmbeddingFunction(
        api_key=provider.api_key,
        model_name=provider.model,
        api_base=provider.base_url,
    )


@register_embedding_converter(EmbeddingCohere)
def _convert_cohere_provider(provider: EmbeddingCohere) -> ChromaEmbeddingFunction:
    from chromadb.utils.embedding_functions import CohereEmbeddingFunction

    if os.getenv("CHROMA_COHERE_API_KEY"):
        return CohereEmbeddingFunction(model_name=provider.model)
    if os.getenv("COHERE_API_KEY"):
        return CohereEmbeddingFunction(
            model_name=provider.model,
            api_key_env_var="COHERE_API_KEY",
        )
    if provider.api_key is None or provider.api_key == os.getenv("CO_API_KEY"):
        return CohereEmbeddingFunction(
            model_name=provider.model,
            api_key_env_var="CO_API_KEY",
        )
    return CohereEmbeddingFunction(
        api_key=provider.api_key,
        model_name=provider.model,
    )


def coerce_chroma_embedding_function(
    embed: Optional[ChromaEmbedding],
) -> Optional[ChromaEmbeddingFunction]:
    if embed is None:
        return None
    if isinstance(embed, EmbeddingProvider):
        return to_chroma_embedding_function(embed)
    if _EmbeddingFunctionBase is not None and isinstance(embed, _EmbeddingFunctionBase):
        return cast(ChromaEmbeddingFunction, embed)
    return cast(ChromaEmbeddingFunction, embed)
