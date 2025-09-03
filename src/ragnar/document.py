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
    pass


@dataclass
class Chunk(ABC):
    content: str


DocType = TypeVar("DocType", bound=Document)
ChunkType = TypeVar("ChunkType", bound=Chunk)


@dataclass
class ChunkedDocument(Generic[DocType, ChunkType]):
    document: DocType
    chunks: list[ChunkType]


@dataclass
class MarkdownChunk(Chunk):
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
        self.chunk_id = chunk_id
        self.parent_doc = parent_doc
        self.start = start
        self.end = end
        self.context = context
        self.content = self.parent_doc.content[self.start : self.end]
