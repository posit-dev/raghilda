from typing import Any, Optional, Sequence

from ._embedding import (
    EmbeddingProvider,
    EmbedInputType,
    register_embedding_provider,
)


@register_embedding_provider("EmbeddingSentenceTransformers")
class EmbeddingSentenceTransformers(EmbeddingProvider):
    """
    Creates an embedding function provider backed by sentence-transformers models.
    Implements the [EmbeddingProvider](`raghilda.EmbeddingProvider`) interface.

    This provider runs models locally using the `sentence-transformers` library,
    enabling offline/private embedding without external API calls.

    Parameters
    ----------
    model
        The sentence-transformers model to use. Default is "all-MiniLM-L6-v2".
        Any model from the Hugging Face Hub that is compatible with
        sentence-transformers can be used.
    device
        The device to run the model on (e.g., "cpu", "cuda", "mps"). If None,
        sentence-transformers will auto-detect the best available device.
    batch_size
        The number of texts to process in each batch.
    prompts
        Optional mapping from `EmbedInputType` to a prefix string to prepend
        to each text before encoding. This is useful for models that require
        task-specific prefixes (e.g., nomic-embed-text uses "search_query: "
        and "search_document: ").

    Examples
    --------
    Install raghilda with sentence-transformers support:

    ```bash
    pip install raghilda[sentence-transformers]
    ```

    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingSentenceTransformers

    provider = EmbeddingSentenceTransformers(model="all-MiniLM-L6-v2")
    embeddings = provider.embed(["hello world", "testing embeddings"])
    print(len(embeddings))
    print(len(embeddings[0]))  # Dimension of the embedding
    ```

    For models that use task-specific prefixes:

    ```{python}
    #| eval: false
    from raghilda.embedding import EmbeddingSentenceTransformers, EmbedInputType

    provider = EmbeddingSentenceTransformers(
        model="nomic-ai/nomic-embed-text-v1.5",
        prompts={
            EmbedInputType.QUERY: "search_query: ",
            EmbedInputType.DOCUMENT: "search_document: ",
        },
    )
    # Queries get "search_query: " prepended automatically
    query_emb = provider.embed(["Who is Laurens van Der Maaten?"], EmbedInputType.QUERY)
    # Documents get "search_document: " prepended automatically
    doc_emb = provider.embed(["TSNE is a dimensionality reduction algorithm"])
    ```
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
        prompts: Optional[dict[EmbedInputType, str]] = None,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = model
        self.device = device
        self.batch_size = batch_size
        self.prompts = prompts

        self.model_instance = SentenceTransformer(model, device=device)

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "type": "EmbeddingSentenceTransformers",
            "model": self.model,
            "batch_size": self.batch_size,
        }
        if self.device is not None:
            config["device"] = self.device
        if self.prompts is not None:
            config["prompts"] = {k.value: v for k, v in self.prompts.items()}
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingSentenceTransformers":
        prompts = None
        if "prompts" in config:
            prompts = {EmbedInputType(k): v for k, v in config["prompts"].items()}
        return cls(
            model=config.get("model", "all-MiniLM-L6-v2"),
            device=config.get("device"),
            batch_size=config.get("batch_size", 64),
            prompts=prompts,
        )

    def embed(
        self,
        x: Sequence[str],
        input_type: EmbedInputType = EmbedInputType.DOCUMENT,
    ) -> Sequence[Sequence[float]]:
        if isinstance(x, str):
            raise TypeError("Input must be a sequence of strings, not a single string.")

        if len(x) == 0:
            return []

        # Check for empty strings
        empty_indices = [i for i, text in enumerate(x) if not text or not text.strip()]
        if empty_indices:
            raise ValueError(
                f"Input contains empty strings at indices: {empty_indices}. "
                "Empty strings cannot be embedded."
            )

        texts = list(x)
        if self.prompts and input_type in self.prompts:
            prefix = self.prompts[input_type]
            texts = [prefix + t for t in texts]

        embeddings = self.model_instance.encode(
            texts, batch_size=self.batch_size, show_progress_bar=False
        )
        return embeddings.tolist()
