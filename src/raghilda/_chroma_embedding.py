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
)

from ._embedding import (
    EmbeddingCohere,
    EmbeddingOpenAI,
    EmbeddingProvider,
    EmbedInputType,
    embedding_from_config,
)

if TYPE_CHECKING:
    from chromadb.api.types import (  # pyright: ignore[reportMissingImports]
        Documents,
        EmbeddingFunction,
    )

    ChromaEmbeddingFunction: TypeAlias = EmbeddingFunction[Documents]
else:
    ChromaEmbeddingFunction = Any

ChromaEmbeddingInput: TypeAlias = EmbeddingProvider | ChromaEmbeddingFunction

try:
    import chromadb  # pyright: ignore[reportMissingImports]
    import chromadb.utils.embedding_functions  # pyright: ignore[reportMissingImports]
except ImportError:
    chromadb = None

if chromadb is not None:

    class _ChromaEmbeddingAdapter(chromadb.EmbeddingFunction):
        """Adapter to use any raghilda EmbeddingProvider with ChromaDB.

        This adapter wraps a raghilda `EmbeddingProvider` to make it compatible with
        ChromaDB's `EmbeddingFunction` protocol, including serialization support.
        Use this for custom embedding providers that don't have a native ChromaDB equivalent.

        The adapter is automatically used when passing an `EmbeddingProvider` to
        `ChromaDBStore.create()` or `connect()` if no converter registration is found.

        Note: This adapter stores the provider config for serialization, but cross-language
        compatibility (e.g., TypeScript) is not supported since the provider is Python-only.

        Parameters
        ----------
        provider
            A raghilda EmbeddingProvider instance.

        Examples
        --------
        ```{python}
        #| eval: false
        from raghilda.embedding import EmbeddingOpenAI
        from raghilda.store import ChromaDBStore

        provider = EmbeddingOpenAI(model="text-embedding-3-small")
        store = ChromaDBStore.create(location="my_store", name="docs", embed=provider)
        ```
        """

        def __init__(self, provider: EmbeddingProvider) -> None:
            self._provider = provider

        def __call__(self, input: Sequence[str]) -> list[Any]:
            """Generate embeddings for documents.

            This method is called by ChromaDB when adding/upserting documents.
            """
            import numpy as np

            embeddings = self._provider.embed(list(input), EmbedInputType.DOCUMENT)
            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        def embed_query(self, input: Sequence[str]) -> list[Any]:
            """Generate embeddings for queries.

            This method is called by ChromaDB when querying the collection.
            """
            import numpy as np

            embeddings = self._provider.embed(list(input), EmbedInputType.QUERY)
            return [np.array(emb, dtype=np.float32) for emb in embeddings]

        @staticmethod
        def name() -> str:
            """Return the adapter registration name used by ChromaDB."""
            return "raghilda_embedding_adapter"

        def get_config(self) -> dict[str, Any]:
            """Return configuration for serialization.

            The config includes the wrapped provider's config so it can be restored.
            """
            return {"provider_config": self._provider.get_config()}

        @staticmethod
        def build_from_config(config: dict[str, Any]) -> "_ChromaEmbeddingAdapter":
            """Restore the adapter from a configuration dict.

            This reconstructs both the adapter and the wrapped provider.
            """
            provider_config = config.get("provider_config", {})
            provider = embedding_from_config(provider_config)
            return _ChromaEmbeddingAdapter(provider)

    chromadb.utils.embedding_functions.register_embedding_function(
        _ChromaEmbeddingAdapter
    )
    ChromaEmbeddingAdapter = _ChromaEmbeddingAdapter

    @singledispatch
    def to_chroma_embedding_function(
        provider: EmbeddingProvider,
    ) -> ChromaEmbeddingFunction:
        """Convert a provider to a ChromaDB embedding function.

        Parameters
        ----------
        provider
            A raghilda embedding provider.

        Returns
        -------
        EmbeddingFunction[Documents]
            A ChromaDB-compatible embedding function.
        """
        return _ChromaEmbeddingAdapter(provider)
else:

    class _MissingChromaEmbeddingAdapter:
        def __init__(self, provider: EmbeddingProvider) -> None:
            raise ModuleNotFoundError(
                "ChromaDB is required to use ChromaDBStore. Install with `pip install chromadb`."
            )

    ChromaEmbeddingAdapter = _MissingChromaEmbeddingAdapter

    @singledispatch
    def to_chroma_embedding_function(
        provider: EmbeddingProvider,
    ) -> ChromaEmbeddingFunction:
        """Convert a provider to a ChromaDB embedding function.

        Parameters
        ----------
        provider
            A raghilda embedding provider.

        Returns
        -------
        EmbeddingFunction[Documents]
            A ChromaDB-compatible embedding function.
        """
        raise ModuleNotFoundError(
            "ChromaDB is required to use ChromaDBStore. Install with `pip install chromadb`."
        )


P = TypeVar("P", bound=EmbeddingProvider)


def register_provider_converter(
    provider_type: type[P],
) -> Callable[
    [Callable[[P], ChromaEmbeddingFunction]],
    Callable[[P], ChromaEmbeddingFunction],
]:
    """Register a provider->Chroma embedding converter.

    The converter is used by `ChromaDBStore.create(..., embed=...)` and
    `ChromaDBStore.connect(..., embed=...)` when `embed` is an instance of
    `provider_type`.
    """
    assert issubclass(provider_type, EmbeddingProvider)

    def decorator(
        converter: Callable[[P], ChromaEmbeddingFunction],
    ) -> Callable[[P], ChromaEmbeddingFunction]:
        to_chroma_embedding_function.register(provider_type)(converter)
        return converter

    return decorator


@register_provider_converter(EmbeddingOpenAI)
def _convert_openai_provider(provider: EmbeddingOpenAI) -> ChromaEmbeddingFunction:
    from chromadb.utils.embedding_functions import (  # pyright: ignore[reportMissingImports]
        OpenAIEmbeddingFunction,
    )

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


@register_provider_converter(EmbeddingCohere)
def _convert_cohere_provider(provider: EmbeddingCohere) -> ChromaEmbeddingFunction:
    from chromadb.utils.embedding_functions import (  # pyright: ignore[reportMissingImports]
        CohereEmbeddingFunction,
    )

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
    embed: Optional[ChromaEmbeddingInput],
) -> Optional[ChromaEmbeddingFunction]:
    """Coerce embed input to a ChromaDB embedding function when needed."""
    if embed is None:
        return None
    if isinstance(embed, EmbeddingProvider):
        return to_chroma_embedding_function(embed)
    return embed
