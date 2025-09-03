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
    origin: Optional[str]
    content: str


@dataclass
class Chunk(ABC):
    content: str


_DocType = TypeVar("DocType", bound=Document)
_ChunkType = TypeVar("ChunkType", bound=Chunk)


@dataclass
class ChunkedDocument(Generic[_DocType, _ChunkType]):
    document: _DocType
    chunks: list[_ChunkType]


@dataclass
class LazyMarkdownChunk(Chunk):
    parent_doc: MarkdownDocument
    chunk_id: int
    start: int
    end: int
    context: Optional[str] = None

    def __init__(
        self,
        chunk_id: int,
        parent_doc: MarkdownDocument,
        start: int,
        end: int,
        context: Optional[str] = None,
    ):
        self.parent_doc = parent_doc
        self.start = start
        self.end = end
        self.context = context
        self.chunk_id = chunk_id

    @property
    def content(self) -> str:
        return self.parent_doc.content[self.start : self.end]
