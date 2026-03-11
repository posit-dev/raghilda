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
    """

    def __init__(
        self,
        model: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = model
        self.device = device
        self.batch_size = batch_size

        self.model_instance = SentenceTransformer(model, device=device)

    def get_config(self) -> dict[str, Any]:
        config: dict[str, Any] = {
            "type": "EmbeddingSentenceTransformers",
            "model": self.model,
            "batch_size": self.batch_size,
        }
        if self.device is not None:
            config["device"] = self.device
        return config

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "EmbeddingSentenceTransformers":
        return cls(
            model=config.get("model", "all-MiniLM-L6-v2"),
            device=config.get("device"),
            batch_size=config.get("batch_size", 64),
        )

    def embed(
        self,
        x: Sequence[str],
        input_type: EmbedInputType = EmbedInputType.DOCUMENT,
    ) -> Sequence[Sequence[float]]:
        # Note: sentence-transformers doesn't differentiate between query and document
        # embeddings by default, so input_type is accepted but ignored.
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

        embeddings = self.model_instance.encode(
            list(x), batch_size=self.batch_size, show_progress_bar=False
        )
        return embeddings.tolist()
