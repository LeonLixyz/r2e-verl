from pathlib import Path

import pytest

from relace_agent.retrieval import FileChunk

COMMON_TESTS = [
    pytest.param(
        "Short text.",
        id="short_text",
    ),
    pytest.param(
        "Many lines of text.\n" * FileChunk.max_chars,
        id="many_lines",
    ),
    pytest.param(
        "One line that exceeds the maximum chunk size " * FileChunk.max_chars,
        id="long_line",
    ),
    pytest.param(
        (f"Short line\n{'Long line ' * (FileChunk.max_chars)}\n") * 10,
        id="mixed_lines",
    ),
]


class TestFileChunk:
    @pytest.mark.parametrize("content", COMMON_TESTS)
    def test_from_content(self, content: str) -> None:
        chunk = FileChunk.from_content("test.txt", content)
        assert chunk.file_path == "test.txt"
        assert chunk.content == content
        assert chunk.content_hash

    @pytest.mark.parametrize("text", COMMON_TESTS)
    def test_iter_text(self, text: str) -> None:
        chunks = list(FileChunk.iter_text("test.txt", text))
        for i, chunk in enumerate(chunks):
            assert chunk.file_path == "test.txt"
            assert len(chunk.content) <= chunk.max_chars, (
                "Chunks should not exceed max_chars"
            )
            if 0 < i < len(chunks) - 1:
                assert len(chunk.content) >= chunk.min_chars, (
                    "Intermediate chunks should not be smaller than min_chars"
                )

        chunks_reconstructed = "".join(chunk.content for chunk in chunks)
        assert chunks_reconstructed == text, (
            "Chunks should reconstruct the original text"
        )

        ids = {chunk.id for chunk in chunks}
        assert len(ids) == len(chunks), "Chunks should have unique IDs"

    @pytest.mark.parametrize("text", COMMON_TESTS)
    @pytest.mark.asyncio
    async def test_iter_file(self, text: str, tmp_path: Path) -> None:
        file_path = tmp_path / "test.txt"
        file_path.write_text(text)

        chunks = [
            chunk async for chunk in FileChunk.iter_file(file_path, root_path=tmp_path)
        ]
        for i, chunk in enumerate(chunks):
            assert chunk.file_path == "test.txt"
            assert len(chunk.content) <= chunk.max_chars, (
                "Chunks should not exceed max_chars"
            )
            if 0 < i < len(chunks) - 1:
                assert len(chunk.content) >= chunk.min_chars, (
                    "Intermediate chunks should not be smaller than min_chars"
                )

        chunks_reconstructed = "".join(chunk.content for chunk in chunks)
        assert chunks_reconstructed == text, (
            "Chunks should reconstruct the original text"
        )

        ids = {chunk.id for chunk in chunks}
        assert len(ids) == len(chunks), "Chunks should have unique IDs"
