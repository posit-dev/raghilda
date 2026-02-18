from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from ._attributes import AttributeSpec, AttributeType

if TYPE_CHECKING:
    from .embedding import EmbeddingProvider


class AttributesStoreMetadata(Protocol):
    """Shared metadata contract for stores that expose typed attributes.

    This protocol centralizes the metadata fields consumed by store
    implementations when inserting, validating, filtering, and returning
    attribute values.

    Attributes
    ----------
    name
        Stable store identifier.
    title
        Human-readable store title used for display.
    attributes
        Canonical attribute declarations (`AttributeSpec` per key). This is the
        source of truth persisted in store metadata.
    """

    name: str
    title: str
    attributes: dict[str, AttributeSpec]

    @property
    def attributes_spec(self) -> dict[str, AttributeSpec]:
        """Return full attribute declarations used for validation and merging.

        Implementations should return the same semantic information as
        ``attributes`` (including required/nullable/default rules), even if the
        underlying storage differs.
        """
        ...

    @property
    def attributes_schema(self) -> dict[str, AttributeType]:
        """Return type-only attribute mapping used for output/filtering surfaces.

        This is a projection of ``attributes_spec`` that keeps only
        ``AttributeType`` values. Stores use it to shape returned
        ``chunk.attributes``, determine filterable user attribute keys, and
        document the attributes available to downstream prompt/context assembly.
        """
        ...


class EmbeddedAttributesStoreMetadata(AttributesStoreMetadata, Protocol):
    """Metadata contract for stores that also carry an embedding provider.

    Attributes
    ----------
    embed
        Embedding provider configuration restored with the store metadata. Used
        by retrieval paths that need to embed queries/documents.
    """

    embed: EmbeddingProvider | None


def attributes_schema_from_spec(
    attributes: dict[str, AttributeSpec],
) -> dict[str, AttributeType]:
    return {key: spec.attribute_type for key, spec in attributes.items()}
