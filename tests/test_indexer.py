"""Tests for the indexer module."""

import tiktoken

from obsidian_rag.indexer import _OPENAI_MAX_TOKENS, _truncate_for_embedding, chunk_markdown, parse_frontmatter


class TestParseFrontmatter:
    def test_no_frontmatter(self):
        content = "# Hello\n\nThis is content."
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}
        assert body == content

    def test_with_frontmatter(self):
        content = """---
title: Test Note
tags:
  - test
  - example
---

# Hello

This is content."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter["title"] == "Test Note"
        assert frontmatter["tags"] == ["test", "example"]
        assert body.startswith("# Hello")

    def test_invalid_yaml(self):
        content = """---
invalid: yaml: content
---

Content here."""
        frontmatter, body = parse_frontmatter(content)
        assert frontmatter == {}


class TestChunkMarkdown:
    def test_single_chunk_no_headings(self):
        """Short content without headings becomes one chunk."""
        content = "This is a simple note without headings."
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) == 1
        assert "simple note" in chunks[0].content

    def test_splits_on_headings(self):
        """Content with headings gets split at heading boundaries when large enough."""
        # Each section needs enough content to exceed chunk_size when combined
        section_text = "This is filler content for the section. " * 50
        content = f"""## First Section

{section_text}

## Second Section

{section_text}"""
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) >= 2

    def test_preserves_file_path(self):
        content = "## Test\n\nContent here that is long enough."
        chunks = chunk_markdown(content, "notes/test.md")
        assert all(c.file_path == "notes/test.md" for c in chunks)

    def test_extracts_frontmatter(self):
        """Frontmatter is parsed and stored in metadata, not in chunk content."""
        content = """---
tags:
  - project
---

## Section

Actual content here."""
        chunks = chunk_markdown(content, "test.md")
        assert len(chunks) >= 1
        # Frontmatter should not appear in chunk content
        assert "---" not in chunks[0].content or "tags" not in chunks[0].content

    def test_assigns_type_metadata(self):
        """Chunks get type metadata based on file path."""
        daily = chunk_markdown("Some content here.", "Daily Notes/2026-01-01.md")
        note = chunk_markdown("Some content here.", "Projects/foo.md")
        assert daily[0].metadata["type"] == "daily"
        assert note[0].metadata["type"] == "note"

    def test_empty_content_returns_empty(self):
        """Empty or whitespace-only content returns no chunks."""
        assert chunk_markdown("", "test.md") == []
        assert chunk_markdown("   \n\n  ", "test.md") == []


class TestTruncateForEmbedding:
    def test_short_text_is_unchanged(self):
        text = "A short note that fits comfortably under the limit."
        assert _truncate_for_embedding(text) == text

    def test_long_text_is_cut_to_token_limit(self):
        enc = tiktoken.encoding_for_model("text-embedding-3-small")
        text = "word " * (_OPENAI_MAX_TOKENS * 2)
        truncated = _truncate_for_embedding(text)
        assert len(enc.encode(truncated)) == _OPENAI_MAX_TOKENS
        assert text.startswith(truncated)

    def test_custom_limit_uses_model_encoding(self):
        enc = tiktoken.get_encoding("cl100k_base")
        text = "alpha beta gamma delta epsilon zeta eta theta"
        truncated = _truncate_for_embedding(text, max_tokens=3, model="unknown-model")
        assert enc.encode(truncated) == enc.encode(text)[:3]

    def test_literal_special_token_text_is_counted_as_plain_text(self):
        """Notes quoting prompt markup like <|endoftext|> must not be rejected by tiktoken."""
        text = "The prompt ends with <|endoftext|> and continues."
        assert _truncate_for_embedding(text) == text
        long_text = ("<|endoftext|> " * (_OPENAI_MAX_TOKENS * 2)).strip()
        enc = tiktoken.get_encoding("cl100k_base")
        truncated = _truncate_for_embedding(long_text)
        assert len(enc.encode(truncated, disallowed_special=())) == _OPENAI_MAX_TOKENS

    def test_skips_the_tokenizer_when_the_text_fits_in_bytes(self, monkeypatch):
        """Every tiktoken token covers at least one byte, so a text under the limit in bytes cannot exceed it."""

        def must_not_tokenize(*_args, **_kwargs):
            raise AssertionError("tokenizer must not run for text under the byte limit")

        monkeypatch.setattr(tiktoken, "encoding_for_model", must_not_tokenize)
        text = "é" * (_OPENAI_MAX_TOKENS // 2) + "a"  # 2 bytes per "é" plus 1: exactly _OPENAI_MAX_TOKENS bytes
        assert len(text.encode("utf-8")) == _OPENAI_MAX_TOKENS
        assert _truncate_for_embedding(text) == text

    def test_tokenizes_when_the_text_exceeds_the_byte_limit_but_not_the_token_limit(self):
        text = "a" * (_OPENAI_MAX_TOKENS + 1)  # more bytes than the limit, far fewer tokens
        assert _truncate_for_embedding(text) == text
