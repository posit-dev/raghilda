from abc import ABC
from dataclasses import dataclass
from typing import Optional, TypeVar, Generic


@dataclass
class Document(ABC):
    origin: Optional[str]
    # TODO: currently only text content, but could be an image, audio etc for multimodal models
    content: str


@dataclass
class MarkdownDocument(Document):
    doc_id: Optional[int] = None


@dataclass
class Chunk(ABC):
    content: str


@dataclass
class MarkdownChunk(Chunk):
    start: int
    end: int
    doc_id: Optional[int] = None
    chunk_id: Optional[int] = None
    context: Optional[str] = None


@dataclass
class Metric:
    name: str
    value: float


@dataclass
class RetrievedChunk(Chunk):
    metrics: list[Metric]


@dataclass
class RetrievedMarkdownChunk(MarkdownChunk, RetrievedChunk):
    pass


DocType = TypeVar("DocType", bound=Document)
ChunkType = TypeVar("ChunkType", bound=Chunk)


@dataclass
class ChunkedDocument(Generic[DocType, ChunkType]):
    document: DocType
    chunks: list[ChunkType]
