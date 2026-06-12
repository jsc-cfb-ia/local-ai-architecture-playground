import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "de",
    "del",
    "el",
    "en",
    "es",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "la",
    "las",
    "los",
    "of",
    "on",
    "or",
    "para",
    "que",
    "should",
    "the",
    "to",
    "un",
    "una",
    "what",
    "with",
    "y",
}

TOPIC_KEYWORDS = {
    "aws": {"aws", "cloud", "serverless"},
    "event-driven": {
        "event",
        "eventbridge",
        "lambda",
        "sns",
        "sqs",
    },
    "graphql": {"appsync", "graphql", "mutation", "query", "subscription"},
    "orchestration": {
        "orchestration",
        "state",
        "step",
        "workflow",
    },
}


@dataclass(frozen=True)
class KnowledgeChunk:
    """A searchable piece of a local knowledge document."""

    source: str
    topic: str
    chunk_index: int
    content: str


@dataclass(frozen=True)
class SearchResult:
    """A ranked retrieval result with explainable scoring."""

    chunk: KnowledgeChunk
    score: float
    matched_terms: tuple[str, ...]


def tokenize(text: str) -> list[str]:
    """Convert text into normalized tokens while preserving frequency."""

    return re.findall(r"[a-z0-9]+", text.lower())


def meaningful_tokens(text: str) -> list[str]:
    """Remove common words that do not indicate document relevance."""

    return [
        token
        for token in tokenize(text)
        if token not in STOP_WORDS and len(token) > 1
    ]


def classify_topic(file_name: str, content: str) -> str:
    """Classify a document using transparent keyword matching."""

    file_tokens = set(tokenize(file_name))
    content_tokens = set(tokenize(content))
    topic_scores = {
        topic: (
            len(content_tokens.intersection(keywords))
            + len(file_tokens.intersection(keywords)) * 2
        )
        for topic, keywords in TOPIC_KEYWORDS.items()
    }
    best_topic, best_score = max(
        topic_scores.items(),
        key=lambda item: item[1],
    )

    return best_topic if best_score > 0 else "general"


def chunk_text(
    content: str,
    max_words: int = 120,
    overlap_words: int = 20,
) -> list[str]:
    """Split text into overlapping word windows."""

    if max_words <= 0:
        raise ValueError("max_words must be greater than zero")

    if overlap_words < 0 or overlap_words >= max_words:
        raise ValueError(
            "overlap_words must be non-negative and smaller than max_words"
        )

    words = content.split()

    if not words:
        return []

    step = max_words - overlap_words
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunks.append(" ".join(words[start:end]))

        if end == len(words):
            break

        start += step

    return chunks


def read_knowledge_chunks(
    knowledge_dir: Path = KNOWLEDGE_DIR,
    max_words: int = 120,
    overlap_words: int = 20,
) -> list[KnowledgeChunk]:
    """Read local files and turn them into searchable chunks."""

    chunks = []

    for file_path in sorted(knowledge_dir.glob("*.txt")):
        content = file_path.read_text(encoding="utf-8").strip()
        topic = classify_topic(file_path.stem, content)

        for index, chunk in enumerate(
            chunk_text(content, max_words, overlap_words),
            start=1,
        ):
            chunks.append(
                KnowledgeChunk(
                    source=file_path.name,
                    topic=topic,
                    chunk_index=index,
                    content=chunk,
                )
            )

    return chunks


def calculate_score(
    query: str,
    chunk: KnowledgeChunk,
    context: str = "",
) -> tuple[float, tuple[str, ...]]:
    """Score a chunk, favoring the current question over old context."""

    query_counts = Counter(meaningful_tokens(query))
    context_counts = Counter(meaningful_tokens(context))
    content_counts = Counter(meaningful_tokens(chunk.content))
    source_tokens = set(meaningful_tokens(chunk.source))
    topic_tokens = set(meaningful_tokens(chunk.topic))

    matched_terms = tuple(
        sorted(set(query_counts).intersection(content_counts))
    )
    source_matches = set(query_counts).intersection(source_tokens)
    topic_matches = set(query_counts).intersection(topic_tokens)
    query_score = sum(
        min(count, content_counts[token]) * 2.0
        for token, count in query_counts.items()
    )
    context_score = sum(
        min(count, content_counts[token]) * 0.35
        for token, count in context_counts.items()
    )
    source_score = len(source_matches) * 3.0
    topic_score = len(topic_matches) * 2.0
    coverage_score = (
        len(matched_terms) / len(query_counts)
        if query_counts
        else 0.0
    )

    return (
        query_score
        + context_score
        + source_score
        + topic_score
        + coverage_score,
        matched_terms,
    )


def search(
    query: str,
    context: str = "",
    limit: int = 3,
    knowledge_dir: Path = KNOWLEDGE_DIR,
) -> list[SearchResult]:
    """Return ranked local knowledge chunks."""

    if limit <= 0:
        return []

    results = []
    query_tokens = set(meaningful_tokens(query))

    for chunk in read_knowledge_chunks(knowledge_dir):
        score, matched_terms = calculate_score(query, chunk, context)
        metadata_tokens = set(
            meaningful_tokens(f"{chunk.source} {chunk.topic}")
        )
        has_direct_query_match = bool(
            matched_terms or query_tokens.intersection(metadata_tokens)
        )

        if score > 0 and has_direct_query_match:
            results.append(
                SearchResult(
                    chunk=chunk,
                    score=score,
                    matched_terms=matched_terms,
                )
            )

    return sorted(
        results,
        key=lambda result: (
            -result.score,
            result.chunk.source,
            result.chunk.chunk_index,
        ),
    )[:limit]


def search_best_match(
    query: str,
    context: str = "",
    knowledge_dir: Path = KNOWLEDGE_DIR,
) -> str:
    """Format the best result for the terminal assistant."""

    results = search(
        query=query,
        context=context,
        limit=1,
        knowledge_dir=knowledge_dir,
    )

    if not results:
        return "I could not find relevant local knowledge for that topic."

    result = results[0]
    chunk = result.chunk

    return (
        f"Source: {chunk.source}\n"
        f"Topic: {chunk.topic}\n"
        f"Chunk: {chunk.chunk_index}\n"
        f"Score: {result.score:.2f}\n\n"
        f"{chunk.content}"
    )
