from dataclasses import dataclass, field
from typing import Optional
from chonkie.types import Chunk, Document


@dataclass
class MarkdownDocument(Document):
    origin: Optional[str] = None


@dataclass
class Metric:
    name: str
    value: float


@dataclass
class RetrievedChunk(Chunk):
    metrics: list[Metric] = field(default_factory=list)
