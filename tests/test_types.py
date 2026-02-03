import pytest
from raghilda._types import (
    Chunk,
    ChunkLike,
    IntoChunk,
    Document,
    DocumentLike,
)


class TestChunkLikeProtocol:
    def test_chunk_satisfies_chunk_like(self):
        chunk = Chunk(text="hello", start_index=0, end_index=5, token_count=5)
        assert isinstance(chunk, ChunkLike)

    def test_custom_class_satisfies_chunk_like(self):
        class MyChunk:
            text = "hello"
            start_index = 0
            end_index = 5
            token_count = 5

        assert isinstance(MyChunk(), ChunkLike)

    def test_partial_implementation_fails_chunk_like(self):
        class PartialChunk:
            text = "hello"
            start_index = 0
            # missing end_index and token_count

        assert not isinstance(PartialChunk(), ChunkLike)

    def test_dict_is_not_chunk_like(self):
        d = {"text": "hello", "start_index": 0, "end_index": 5, "token_count": 5}
        assert not isinstance(d, ChunkLike)

    def test_from_any_with_chunk_like(self):
        class MyChunk:
            text = "hello"
            start_index = 0
            end_index = 5
            token_count = 5

        result = Chunk.from_any(MyChunk())
        assert isinstance(result, Chunk)
        assert result.text == "hello"
        assert result.start_index == 0

    def test_from_any_preserves_context_if_present(self):
        class MyChunkWithContext:
            text = "hello"
            start_index = 0
            end_index = 5
            token_count = 5
            context = "# Header"

        result = Chunk.from_any(MyChunkWithContext())
        assert result.context == "# Header"

    def test_from_any_context_is_none_if_missing(self):
        class MyChunkNoContext:
            text = "hello"
            start_index = 0
            end_index = 5
            token_count = 5

        result = Chunk.from_any(MyChunkNoContext())
        assert result.context is None


class TestIntoChunkProtocol:
    def test_class_with_to_chunk_satisfies_into_chunk(self):
        class Convertible:
            def to_chunk(self) -> Chunk:
                return Chunk(
                    text="converted", start_index=0, end_index=9, token_count=9
                )

        assert isinstance(Convertible(), IntoChunk)

    def test_from_any_calls_to_chunk(self):
        class Convertible:
            def to_chunk(self) -> Chunk:
                return Chunk(
                    text="converted", start_index=0, end_index=9, token_count=9
                )

        result = Chunk.from_any(Convertible())
        assert result.text == "converted"

    def test_to_chunk_wrong_signature_no_return(self):
        """to_chunk() that returns None instead of Chunk."""

        class BadConvertible:
            def to_chunk(self):
                pass  # returns None

        # It satisfies IntoChunk protocol (has the method)
        assert isinstance(BadConvertible(), IntoChunk)
        # But from_any raises TypeError due to runtime validation
        with pytest.raises(TypeError, match="must return a Chunk, got NoneType"):
            Chunk.from_any(BadConvertible())

    def test_to_chunk_wrong_signature_returns_string(self):
        """to_chunk() that returns wrong type."""

        class BadConvertible:
            def to_chunk(self):
                return "not a chunk"

        assert isinstance(BadConvertible(), IntoChunk)
        # from_any raises TypeError due to runtime validation
        with pytest.raises(TypeError, match="must return a Chunk, got str"):
            Chunk.from_any(BadConvertible())

    def test_to_chunk_with_required_args_fails_at_runtime(self):
        """to_chunk() that requires arguments will fail when called."""

        class BadConvertible:
            def to_chunk(self, required_arg):
                return Chunk(
                    text=required_arg, start_index=0, end_index=5, token_count=5
                )

        assert isinstance(BadConvertible(), IntoChunk)
        with pytest.raises(TypeError, match="required_arg"):
            Chunk.from_any(BadConvertible())

    def test_to_chunk_as_property_not_method(self):
        """to_chunk as a property instead of method."""

        class PropertyChunk:
            @property
            def to_chunk(self):
                return Chunk(text="prop", start_index=0, end_index=4, token_count=4)

        # isinstance returns True for properties too (has the attribute)
        assert isinstance(PropertyChunk(), IntoChunk)
        # But from_any raises TypeError because it's not callable
        with pytest.raises(TypeError, match="must be a method"):
            Chunk.from_any(PropertyChunk())


class TestIsinstance:
    def test_isinstance_into_chunk_true_for_to_chunk_method(self):
        class WithToChunk:
            def to_chunk(self):
                pass

        assert isinstance(WithToChunk(), IntoChunk) is True

    def test_isinstance_into_chunk_false_for_chunk_like(self):
        class OnlyChunkLike:
            text = "hello"
            start_index = 0
            end_index = 5
            token_count = 5

        assert isinstance(OnlyChunkLike(), IntoChunk) is False

    def test_isinstance_chunk_like_true_for_attributes(self):
        class WithAttributes:
            text = "hello"
            start_index = 0
            end_index = 5
            token_count = 5

        assert isinstance(WithAttributes(), ChunkLike) is True

    def test_isinstance_chunk_like_false_for_only_to_chunk(self):
        class OnlyToChunk:
            def to_chunk(self):
                pass

        assert isinstance(OnlyToChunk(), ChunkLike) is False

    def test_class_with_both_prefers_to_chunk(self):
        """When a class has both to_chunk() and ChunkLike attributes, to_chunk wins."""

        class Both:
            text = "from_attributes"
            start_index = 0
            end_index = 15
            token_count = 15

            def to_chunk(self) -> Chunk:
                return Chunk(
                    text="from_to_chunk", start_index=0, end_index=13, token_count=13
                )

        result = Chunk.from_any(Both())
        assert result.text == "from_to_chunk"


class TestDocumentProtocols:
    def test_document_satisfies_document_like(self):
        doc = Document(content="hello world")
        assert isinstance(doc, DocumentLike)

    def test_custom_class_satisfies_document_like(self):
        class MyDoc:
            content = "hello"
            chunks = None

        assert isinstance(MyDoc(), DocumentLike)

    def test_from_any_with_document_like(self):
        class MyDoc:
            content = "hello world"
            chunks = None

        result = Document.from_any(MyDoc())
        assert isinstance(result, Document)
        assert result.content == "hello world"

    def test_from_any_converts_chunks(self):
        class MyChunk:
            text = "chunk1"
            start_index = 0
            end_index = 6
            token_count = 6

        class MyDoc:
            content = "chunk1 chunk2"
            chunks = [MyChunk()]

        result = Document.from_any(MyDoc())
        assert len(result.chunks) == 1
        assert isinstance(result.chunks[0], Chunk)
        assert result.chunks[0].text == "chunk1"

    def test_into_document_calls_to_document(self):
        class Convertible:
            def to_document(self) -> Document:
                return Document(content="converted")

        result = Document.from_any(Convertible())
        assert result.content == "converted"


class TestErrorCases:
    def test_from_any_raises_for_invalid_type(self):
        """from_any should raise TypeError for objects that match neither protocol."""

        class Invalid:
            pass

        with pytest.raises(TypeError, match="Cannot convert"):
            Chunk.from_any(Invalid())

    def test_document_from_any_raises_for_invalid_type(self):
        class Invalid:
            pass

        with pytest.raises(TypeError, match="Cannot convert"):
            Document.from_any(Invalid())
