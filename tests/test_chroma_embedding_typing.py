from typing import TYPE_CHECKING, Any, Callable, Sequence, assert_type

if TYPE_CHECKING:
    from chromadb.api.types import Documents, EmbeddingFunction

    from raghilda.embedding import EmbedInputType, EmbeddingProvider
    from raghilda.store import ChromaDBStore

    class _TypecheckProvider(EmbeddingProvider):
        def embed(
            self,
            x: Sequence[str],
            input_type: EmbedInputType = EmbedInputType.DOCUMENT,
        ) -> Sequence[Sequence[float]]:
            return [[1.0] for _ in x]

        def get_config(self) -> dict[str, Any]:
            return {"type": "TypecheckProvider"}

        @classmethod
        def from_config(cls, config: dict[str, Any]) -> "_TypecheckProvider":
            return cls()

    @ChromaDBStore.register_provider_converter(_TypecheckProvider)
    def _typed_converter(
        provider: _TypecheckProvider,
    ) -> EmbeddingFunction[Documents]: ...

    _register = ChromaDBStore.register_provider_converter(_TypecheckProvider)
    assert_type(
        _register,
        Callable[
            [Callable[[_TypecheckProvider], EmbeddingFunction[Documents]]],
            Callable[[_TypecheckProvider], EmbeddingFunction[Documents]],
        ],
    )
    assert_type(
        _typed_converter,
        Callable[[_TypecheckProvider], EmbeddingFunction[Documents]],
    )
