import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import EmbeddingModel, OllamaEmbeddingModel
from app.retriever import (
    KNOWLEDGE_DIR,
    PROJECT_ROOT,
    KnowledgeChunk,
    SearchResult,
    read_knowledge_chunks,
)

CACHE_DIR = PROJECT_ROOT / ".cache" / "embeddings"


@dataclass(frozen=True)
class EmbeddedChunk:
    """A knowledge chunk paired with its embedding vector."""

    chunk: KnowledgeChunk
    embedding: list[float]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""

    if not left or not right or len(left) != len(right):
        return 0.0

    dot_product = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))

    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0

    return dot_product / (left_norm * right_norm)


def chunk_cache_key(chunk: KnowledgeChunk) -> str:
    """Build a stable key for one source chunk."""

    raw_key = (
        f"{chunk.source}:{chunk.topic}:{chunk.chunk_index}:"
        f"{hashlib.sha256(chunk.content.encode('utf-8')).hexdigest()}"
    )

    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def cache_path(model_name: str, cache_dir: Path = CACHE_DIR) -> Path:
    safe_name = model_name.replace("/", "_").replace(":", "_")
    return cache_dir / f"{safe_name}.json"


def load_embedding_cache(path: Path) -> dict[str, list[float]]:
    if not path.exists():
        return {}

    raw_cache = json.loads(path.read_text(encoding="utf-8"))

    return {
        key: [float(value) for value in vector]
        for key, vector in raw_cache.items()
    }


def save_embedding_cache(
    path: Path,
    cache: dict[str, list[float]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(cache, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_embedded_chunks(
    embedding_model: EmbeddingModel,
    knowledge_dir: Path = KNOWLEDGE_DIR,
    cache_dir: Path = CACHE_DIR,
) -> list[EmbeddedChunk]:
    """Embed local knowledge chunks using a persistent cache."""

    chunks = read_knowledge_chunks(knowledge_dir)
    path = cache_path(embedding_model.model, cache_dir)
    cache = load_embedding_cache(path)
    missing_chunks = [
        chunk
        for chunk in chunks
        if chunk_cache_key(chunk) not in cache
    ]

    if missing_chunks:
        embeddings = embedding_model.embed(
            [chunk.content for chunk in missing_chunks]
        )

        for chunk, embedding in zip(missing_chunks, embeddings):
            cache[chunk_cache_key(chunk)] = embedding

        save_embedding_cache(path, cache)

    return [
        EmbeddedChunk(
            chunk=chunk,
            embedding=cache[chunk_cache_key(chunk)],
        )
        for chunk in chunks
    ]


def search_semantic(
    query: str,
    embedding_model: EmbeddingModel | None = None,
    limit: int = 3,
    knowledge_dir: Path = KNOWLEDGE_DIR,
    cache_dir: Path = CACHE_DIR,
) -> list[SearchResult]:
    """Search local knowledge by vector similarity."""

    if limit <= 0:
        return []

    model = embedding_model or OllamaEmbeddingModel()
    query_embedding = model.embed([query])[0]
    embedded_chunks = build_embedded_chunks(
        embedding_model=model,
        knowledge_dir=knowledge_dir,
        cache_dir=cache_dir,
    )
    results = [
        SearchResult(
            chunk=embedded_chunk.chunk,
            score=cosine_similarity(
                query_embedding,
                embedded_chunk.embedding,
            ),
            matched_terms=(),
        )
        for embedded_chunk in embedded_chunks
    ]

    return sorted(
        results,
        key=lambda result: (
            -result.score,
            result.chunk.source,
            result.chunk.chunk_index,
        ),
    )[:limit]
