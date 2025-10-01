from abc import ABC
from dataclasses import dataclass, field
from typing import Optional, TypeVar, Generic
from chonkie.types import Chunk


@dataclass
class Document(ABC):
    origin: Optional[str]
    # TODO: currently only text content, but could be an image, audio etc for multimodal models
    content: str


@dataclass
class MarkdownDocument(Document):
    doc_id: Optional[int] = None


@dataclass
class Metric:
    name: str
    value: float


@dataclass
class RetrievedChunk(Chunk):
    metrics: list[Metric] = field(default_factory=list)


DocType = TypeVar("DocType", bound=Document)
ChunkType = TypeVar("ChunkType", bound=Chunk)


@dataclass
class ChunkedDocument(Generic[DocType, ChunkType]):
    document: DocType
    chunks: list[ChunkType]
