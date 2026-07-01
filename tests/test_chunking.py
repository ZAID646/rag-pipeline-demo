import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app import chunk_text


class TestChunkText:
    def test_empty(self):
        assert chunk_text("") == []
        assert chunk_text("   ") == []

    def test_short_text(self):
        result = chunk_text("Hello world", chunk_size=100)
        assert len(result) == 1
        assert "Hello world" in result[0]

    def test_single_word(self):
        result = chunk_text("Hello", chunk_size=100)
        assert result == ["Hello"]

    def test_split_by_paragraph(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        result = chunk_text(text, chunk_size=50)
        assert len(result) >= 1
        assert all(c.strip() for c in result)

    def test_respects_chunk_size(self):
        text = "word " * 200
        result = chunk_text(text, chunk_size=100)
        for chunk in result:
            assert len(chunk) <= 120  # slight allowance for merge

    def test_no_empty_chunks(self):
        text = "Hello there\n\n\n\nWorld here"
        result = chunk_text(text, chunk_size=50)
        assert all(c.strip() for c in result)

    def test_large_text(self):
        text = "Sentence one. Sentence two. " * 100
        result = chunk_text(text, chunk_size=200)
        assert len(result) > 1
        assert all(len(c) <= 250 for c in result)
