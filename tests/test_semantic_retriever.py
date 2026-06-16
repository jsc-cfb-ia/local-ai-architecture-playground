import tempfile
import unittest
from pathlib import Path

from app.retriever import KnowledgeChunk
from app.semantic_retriever import (
    build_embedded_chunks,
    chunk_cache_key,
    cosine_similarity,
    search_semantic,
)


class FakeEmbeddingModel:
    model = "fake-embedding-model"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        vectors = []

        for text in texts:
            lower_text = text.lower()

            if "graphql" in lower_text or "api" in lower_text:
                vectors.append([1.0, 0.0, 0.0])
            elif "workflow" in lower_text or "orchestration" in lower_text:
                vectors.append([0.0, 1.0, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])

        return vectors


class SemanticRetrieverTests(unittest.TestCase):
    def test_cosine_similarity_scores_related_vectors(self) -> None:
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [1.0, 0.0]),
            1.0,
        )
        self.assertAlmostEqual(
            cosine_similarity([1.0, 0.0], [0.0, 1.0]),
            0.0,
        )

    def test_chunk_cache_key_changes_with_content(self) -> None:
        first = KnowledgeChunk("source.txt", "topic", 1, "alpha")
        second = KnowledgeChunk("source.txt", "topic", 1, "beta")

        self.assertNotEqual(
            chunk_cache_key(first),
            chunk_cache_key(second),
        )

    def test_semantic_search_uses_embedding_similarity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            knowledge_dir = Path(directory) / "knowledge"
            cache_dir = Path(directory) / "cache"
            knowledge_dir.mkdir()
            (knowledge_dir / "api.txt").write_text(
                "GraphQL exposes an API shape for clients.",
                encoding="utf-8",
            )
            (knowledge_dir / "workflow.txt").write_text(
                "Step Functions coordinates workflow orchestration.",
                encoding="utf-8",
            )

            results = search_semantic(
                "client api design",
                embedding_model=FakeEmbeddingModel(),
                knowledge_dir=knowledge_dir,
                cache_dir=cache_dir,
                limit=1,
            )

        self.assertEqual(results[0].chunk.source, "api.txt")

    def test_embedding_cache_prevents_reembedding_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            knowledge_dir = Path(directory) / "knowledge"
            cache_dir = Path(directory) / "cache"
            knowledge_dir.mkdir()
            (knowledge_dir / "api.txt").write_text(
                "GraphQL exposes an API shape for clients.",
                encoding="utf-8",
            )
            model = FakeEmbeddingModel()

            build_embedded_chunks(model, knowledge_dir, cache_dir)
            build_embedded_chunks(model, knowledge_dir, cache_dir)

        self.assertEqual(model.calls, 1)


if __name__ == "__main__":
    unittest.main()
