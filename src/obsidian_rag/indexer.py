"""Markdown parsing, chunking, and embedding generation."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import httpx
import yaml
from chonkie import RecursiveChunker
from chonkie.types.recursive import RecursiveLevel, RecursiveRules


@dataclass
class IndexerConfig:
    """Tunable parameters for indexing, chunking, and search.

    Loaded from ``config.toml`` under an ``[indexer]`` section::

        [indexer]
        preset = "math"

    Any field set explicitly in TOML overrides the preset defaults.
    """

    preset: str = "default"
    chunk_size: int = 1500
    chunk_overlap: int = 0  # reserved for future OverlapRefinery support
    min_characters_per_chunk: int = 50
    heading_split_depth: int = 4
    preserve_latex_blocks: bool = False
    preserve_code_blocks: bool = True
    similarity_threshold: float = 0.10
    default_search_limit: int = 10
    default_similar_limit: int = 5
    default_context_limit: int = 5
    extra_exclude_patterns: list[str] = field(default_factory=list)

    def __post_init__(self):
        if self.preset != "default":
            self._apply_preset_defaults()

    @classmethod
    def _make_defaults(cls, preset: str = "default") -> IndexerConfig:
        """Create a baseline instance with raw defaults (bypassing __post_init__)."""
        obj = object.__new__(cls)
        obj.preset = preset
        obj.chunk_size = 1500
        obj.chunk_overlap = 0  # reserved for future OverlapRefinery support
        obj.min_characters_per_chunk = 50
        obj.heading_split_depth = 4
        obj.preserve_latex_blocks = False
        obj.preserve_code_blocks = True
        obj.similarity_threshold = 0.10
        obj.default_search_limit = 10
        obj.default_similar_limit = 5
        obj.default_context_limit = 5
        obj.extra_exclude_patterns = []
        return obj

    def _apply_preset_defaults(self):
        presets = _PRESETS.get(self.preset)
        if presets is None:
            return
        defaults = self._make_defaults()

        for k, v in presets.items():
            if getattr(self, k) == getattr(defaults, k):
                setattr(self, k, v)

    def to_dict(self) -> dict:
        """Serialize for TOML. Only includes fields that differ from preset defaults."""
        base = self._make_defaults(self.preset)
        if self.preset in _PRESETS:
            for k, v in _PRESETS[self.preset].items():
                setattr(base, k, v)

        out: dict = {"preset": self.preset}
        for k in _SERIALIZABLE_FIELDS:
            val = getattr(self, k)
            if val != getattr(base, k):
                out[k] = val
        return out

    @classmethod
    def from_dict(cls, data: dict) -> IndexerConfig:
        """Deserialize from a TOML ``[indexer]`` dict."""
        preset = data.get("preset", "default")
        cfg = cls(preset=preset)
        for k, v in data.items():
            if k == "preset":
                continue
            if k in _SERIALIZABLE_FIELDS:
                setattr(cfg, k, v)
        return cfg


_SERIALIZABLE_FIELDS = [
    "chunk_size",
    "chunk_overlap",
    "min_characters_per_chunk",
    "heading_split_depth",
    "preserve_latex_blocks",
    "preserve_code_blocks",
    "similarity_threshold",
    "default_search_limit",
    "default_similar_limit",
    "default_context_limit",
    "extra_exclude_patterns",
]

_PRESETS: dict[str, dict] = {
    "math": {
        "chunk_size": 1024,
        "min_characters_per_chunk": 80,
        "heading_split_depth": 3,
        "preserve_latex_blocks": True,
        "preserve_code_blocks": True,
        # Optimized for Qwen3-embedding:4b on Latex Math
        # Latex often returns lower similarity scores for relevant chunks
        "similarity_threshold": 0.10,
    },
    "prose": {
        "chunk_size": 1200,
        "min_characters_per_chunk": 40,
        "heading_split_depth": 4,
        "similarity_threshold": 0.10,
    },
}


@dataclass
class Chunk:
    """A chunk of text from a markdown file."""

    id: str
    content: str
    file_path: str
    heading: str | None
    heading_level: int
    metadata: dict


def parse_frontmatter(content: str) -> tuple[dict, str]:
    """Extract YAML frontmatter from markdown content."""
    if not content.startswith("---"):
        return {}, content

    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}, content

    try:
        frontmatter = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        frontmatter = {}

    return frontmatter, parts[2].strip()


_LATEX_ENV_RE = re.compile(
    r"(\\begin\{[^}]+\}.*?\\end\{[^}]+\})",
    re.DOTALL,
)
_DISPLAY_MATH_RE = re.compile(
    r"(\$\$.*?\$\$)",
    re.DOTALL,
)
_FENCED_CODE_RE = re.compile(
    r"(```[^\n]*\n.*?\n```)",
    re.DOTALL,
)

# U+2028 LINE SEPARATOR — not used in markdown, invisible to the chunker's
# delimiter rules, but preserves the character count so token budgets stay
# accurate.  Restored to \n after chunking.
_NEWLINE_MASK = "\u2028"


def _mask_newlines_in_block(match: re.Match) -> str:
    """Replace newlines inside a matched block so the chunker can't split there."""
    return match.group(0).replace("\n", _NEWLINE_MASK)


def _protect_blocks(body: str, cfg: IndexerConfig) -> str:
    """Mask newlines inside LaTeX/code blocks to prevent mid-block splits.

    Replaces \\n inside matched blocks with a non-delimiter character so the
    chunker sees the real content length but finds no split points inside
    the block.
    """
    if cfg.preserve_latex_blocks:
        body = _LATEX_ENV_RE.sub(_mask_newlines_in_block, body)
        body = _DISPLAY_MATH_RE.sub(_mask_newlines_in_block, body)

    if cfg.preserve_code_blocks:
        body = _FENCED_CODE_RE.sub(_mask_newlines_in_block, body)

    return body


def _restore_blocks(text: str) -> str:
    """Restore masked newlines."""
    return text.replace(_NEWLINE_MASK, "\n")


def _build_rules(cfg: IndexerConfig) -> RecursiveRules:
    """Build RecursiveRules according to heading_split_depth."""
    levels: list[RecursiveLevel] = []

    heading_delimiters = [
        ("\n# ", 1),
        ("\n## ", 2),
        ("\n### ", 3),
        ("\n#### ", 4),
        ("\n##### ", 5),
        ("\n###### ", 6),
    ]
    for delim, depth in heading_delimiters:
        if depth <= cfg.heading_split_depth:
            levels.append(RecursiveLevel(delimiters=delim, include_delim="next"))

    levels.extend(
        [
            RecursiveLevel(delimiters="\n\n"),
            RecursiveLevel(delimiters="\n"),
            RecursiveLevel(delimiters=[". ", "! ", "? "]),
            RecursiveLevel(whitespace=True),
        ]
    )

    return RecursiveRules(levels=levels)


@lru_cache(maxsize=16)
def _get_chunker(
    chunk_size: int, chunk_overlap: int, min_characters_per_chunk: int, heading_split_depth: int
) -> RecursiveChunker:
    """Get or create a cached RecursiveChunker for the given parameters."""
    cfg = IndexerConfig(heading_split_depth=heading_split_depth)
    return RecursiveChunker(
        chunk_size=chunk_size,
        rules=_build_rules(cfg),
        min_characters_per_chunk=min_characters_per_chunk,
    )


_default_config = IndexerConfig()


def chunk_markdown(
    content: str,
    file_path: str,
    config: IndexerConfig | None = None,
) -> list[Chunk]:
    """Split markdown content into chunks using Chonkie RecursiveChunker.

    Args:
        content: Raw markdown content (may include frontmatter)
        file_path: Relative path to the source file
        config: Optional IndexerConfig; uses module default if omitted

    Returns:
        List of Chunk objects
    """
    cfg = config or _default_config
    frontmatter, body = parse_frontmatter(content)

    if not body.strip():
        return []

    protected_body = _protect_blocks(body, cfg)

    chunker = _get_chunker(
        cfg.chunk_size,
        cfg.chunk_overlap,
        cfg.min_characters_per_chunk,
        cfg.heading_split_depth,
    )
    chonkie_chunks = chunker.chunk(protected_body)

    note_type = "daily" if file_path.startswith("Daily Notes/") else "note"

    chunks = []
    for i, cc in enumerate(chonkie_chunks):
        text = _restore_blocks(cc.text).strip()
        if not text:
            continue

        heading = None
        heading_level = 0
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("#"):
                level = len(line) - len(line.lstrip("#"))
                if 1 <= level <= 6 and len(line) > level and line[level] == " ":
                    heading = line[level:].strip()
                    heading_level = level
                break

        chunk_id = _generate_chunk_id(file_path, heading, text, i)
        meta = {**frontmatter, "type": note_type, "file_path": file_path}

        chunks.append(
            Chunk(
                id=chunk_id,
                content=text,
                file_path=file_path,
                heading=heading,
                heading_level=heading_level,
                metadata=meta,
            )
        )

    return chunks


def _generate_chunk_id(
    file_path: str,
    heading: str | None,
    content: str,
    chunk_index: int = 0,
) -> str:
    """Generate a stable ID for a chunk."""
    content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
    key = f"{file_path}:{heading or 'root'}:{content_hash}:{chunk_index}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


_OPENAI_MAX_TOKENS = 8191


def _truncate_for_embedding(
    text: str,
    max_tokens: int = _OPENAI_MAX_TOKENS,
    model: str = "text-embedding-3-small",
) -> str:
    """Truncate text to fit within the model's token limit.

    Uses tiktoken if available (with model-specific encoding); falls back to
    a conservative 4 chars/token estimate.
    """
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        tokens = enc.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return enc.decode(tokens[:max_tokens])
    except ImportError:
        char_limit = max_tokens * 4
        if len(text) <= char_limit:
            return text
        return text[:char_limit]


class OpenAIEmbedder:
    """Generate embeddings using OpenAI API."""

    _MAX_RETRIES = 3
    _RETRY_DELAYS = (1.0, 4.0, 16.0)

    def __init__(self, model: str = "text-embedding-3-small", api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.model = model

    def _call_with_retry(self, texts: list[str]) -> list:
        """Call the embeddings API with exponential backoff on transient errors."""
        import time

        from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

        for attempt in range(self._MAX_RETRIES):
            try:
                return self.client.embeddings.create(input=texts, model=self.model)
            except (RateLimitError, APITimeoutError, APIConnectionError):
                if attempt == self._MAX_RETRIES - 1:
                    raise
                time.sleep(self._RETRY_DELAYS[attempt])
            except APIStatusError as e:
                if e.status_code < 500 or attempt == self._MAX_RETRIES - 1:
                    raise
                time.sleep(self._RETRY_DELAYS[attempt])
        raise RuntimeError("unreachable")

    def embed(self, text: str, task_type: str = "search_document") -> list[float]:
        safe_text = _truncate_for_embedding(text, model=self.model)
        response = self._call_with_retry([safe_text])
        return response.data[0].embedding

    # OpenAI allows max 300k tokens per request; use conservative sub-batches
    _MAX_TEXTS_PER_BATCH = 100

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [_truncate_for_embedding(t, model=self.model) for t in texts]
        if len(safe_texts) <= self._MAX_TEXTS_PER_BATCH:
            response = self._call_with_retry(safe_texts)
            return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
        # Sub-batch to stay under API token limits
        all_embeddings: list[list[float]] = []
        for i in range(0, len(safe_texts), self._MAX_TEXTS_PER_BATCH):
            batch = safe_texts[i : i + self._MAX_TEXTS_PER_BATCH]
            response = self._call_with_retry(batch)
            all_embeddings.extend(item.embedding for item in sorted(response.data, key=lambda x: x.index))
        return all_embeddings

    def close(self):
        pass


class OllamaEmbedder:
    """Generate embeddings using Ollama (local)."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "nomic-embed-text"):
        self.base_url = base_url
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def _get_prefix(self, task_type: str) -> str:
        """Get task-specific prefix for models that support them."""
        model = self.model.lower()
        if "nomic" in model:
            if task_type == "search_document":
                return "search_document: "
            elif task_type == "search_query":
                return "search_query: "
        elif "qwen" in model:
            if task_type == "search_query":
                return "Query: "
        return ""

    def embed(self, text: str, task_type: str = "search_document") -> list[float]:
        prefix = self._get_prefix(task_type)
        response = self.client.post(
            f"{self.base_url}/api/embeddings", json={"model": self.model, "prompt": f"{prefix}{text}"}
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_batch(self, texts: list[str], task_type: str = "search_document") -> list[list[float]]:
        return [self.embed(text, task_type) for text in texts]

    def close(self):
        self.client.close()


class LMStudioEmbedder:
    """Generate embeddings using LM Studio (local, OpenAI-compatible API)."""

    def __init__(self, base_url: str = "http://localhost:1234", model: str = "text-embedding-nomic-embed-text-v1.5"):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.client = httpx.Client(timeout=60.0)

    def _get_prefix(self, task_type: str) -> str:
        model = self.model.lower()
        if "nomic" in model:
            if task_type == "search_document":
                return "search_document: "
            elif task_type == "search_query":
                return "search_query: "
        elif "qwen" in model:
            if task_type == "search_query":
                return "Query: "
        return ""

    def embed(self, text: str, task_type: str = "search_document") -> list[float]:
        prefix = self._get_prefix(task_type)
        response = self.client.post(
            f"{self.base_url}/v1/embeddings", json={"model": self.model, "input": f"{prefix}{text}"}
        )
        response.raise_for_status()
        return response.json()["data"][0]["embedding"]

    def embed_batch(self, texts: list[str], task_type: str = "search_document") -> list[list[float]]:
        prefix = self._get_prefix(task_type)
        prefixed_texts = [f"{prefix}{t}" for t in texts]
        response = self.client.post(
            f"{self.base_url}/v1/embeddings", json={"model": self.model, "input": prefixed_texts}
        )
        response.raise_for_status()
        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]

    def close(self):
        self.client.close()


def is_lmstudio_running(base_url: str = "http://localhost:1234") -> bool:
    """Check if LM Studio server is running."""
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/v1/models")
            return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


def is_ollama_running(base_url: str = "http://localhost:11434") -> bool:
    """Check if Ollama server is running."""
    try:
        with httpx.Client(timeout=2.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            return response.status_code == 200
    except (httpx.RequestError, httpx.TimeoutException):
        return False


def get_lmstudio_models(base_url: str = "http://localhost:1234") -> list[str]:
    """Get list of available embedding models from LM Studio."""
    embedding_keywords = ["embed", "bge", "minilm", "e5", "gte", "instructor"]
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/v1/models")
            if response.status_code != 200:
                return []
            data = response.json()
            models = []
            for model in data.get("data", []):
                model_id = model.get("id", "")
                if any(kw in model_id.lower() for kw in embedding_keywords):
                    models.append(model_id)
            return sorted(models)
    except (httpx.RequestError, httpx.TimeoutException, ValueError):
        return []


def get_ollama_models(base_url: str = "http://localhost:11434") -> list[str]:
    """Get list of available embedding models from Ollama."""
    embedding_keywords = ["embed", "bge", "minilm", "e5", "gte", "instructor", "nomic"]
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(f"{base_url.rstrip('/')}/api/tags")
            if response.status_code != 200:
                return []
            data = response.json()
            models = []
            for model in data.get("models", []):
                model_name = model.get("name", "")
                if any(kw in model_name.lower() for kw in embedding_keywords):
                    models.append(model_name)
            return sorted(models)
    except (httpx.RequestError, httpx.TimeoutException, ValueError):
        return []


Embedder = OpenAIEmbedder | OllamaEmbedder | LMStudioEmbedder


def create_embedder(
    provider: str = "openai",
    model: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
) -> Embedder:
    """Create an embedder instance for the specified provider."""
    if provider == "openai":
        kwargs: dict = {}
        if model:
            kwargs["model"] = model
        if api_key:
            kwargs["api_key"] = api_key
        return OpenAIEmbedder(**kwargs)
    elif provider == "ollama":
        kwargs = {}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return OllamaEmbedder(**kwargs)
    elif provider == "lmstudio":
        kwargs = {}
        if model:
            kwargs["model"] = model
        if base_url:
            kwargs["base_url"] = base_url
        return LMStudioEmbedder(**kwargs)
    else:
        raise ValueError(f"Unknown provider: {provider}. Use 'openai', 'ollama', or 'lmstudio'.")


class VaultIndexer:
    """Index an Obsidian vault."""

    def __init__(
        self,
        vault_path,
        embedder: Embedder,
        exclude_patterns: list[str] | None = None,
        config: IndexerConfig | None = None,
    ):
        self.vault_path = Path(vault_path)
        self.embedder = embedder
        self.config = config or IndexerConfig()
        self.exclude_patterns = exclude_patterns or [
            "attachments/**",
            ".obsidian/**",
            ".trash/**",
            ".venv/**",
            "node_modules/**",
            "__pycache__/**",
            "*.egg-info/**",
            "build/**",
            "dist/**",
            ".git/**",
        ]
        if self.config.extra_exclude_patterns:
            self.exclude_patterns = list(set(self.exclude_patterns + self.config.extra_exclude_patterns))

    def iter_markdown_files(self) -> Iterator[Path]:
        """Iterate over all markdown files in the vault."""
        _excluded_dirs = {
            ".obsidian",
            ".trash",
            ".venv",
            "node_modules",
            "__pycache__",
            ".git",
            "build",
            "dist",
        }
        for md_file in self.vault_path.rglob("*.md"):
            rel_path = md_file.relative_to(self.vault_path)
            skip = False
            for part in rel_path.parts:
                if part in _excluded_dirs or part.endswith(".egg-info"):
                    skip = True
                    break
            if not skip:
                for pattern in self.exclude_patterns:
                    if rel_path.match(pattern):
                        skip = True
                        break
            if not skip:
                yield md_file

    def index_file(self, file_path: Path) -> list[tuple[Chunk, list[float]]]:
        """Index a single file, returning chunks with embeddings."""
        content = file_path.read_text(encoding="utf-8")
        rel_path = str(file_path.relative_to(self.vault_path))
        chunks = chunk_markdown(content, rel_path, config=self.config)
        if not chunks:
            return []
        embeddings = self.embedder.embed_batch([c.content for c in chunks])
        return list(zip(chunks, embeddings, strict=False))
