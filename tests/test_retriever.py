import tempfile
import unittest
from pathlib import Path

from app.retriever import (
    chunk_text,
    read_knowledge_chunks,
    search,
    search_best_match,
)


class ChunkTextTests(unittest.TestCase):
    def test_chunks_include_configured_overlap(self) -> None:
        content = "one two three four five six seven"

        chunks = chunk_text(content, max_words=4, overlap_words=2)

        self.assertEqual(
            chunks,
            [
                "one two three four",
                "three four five six",
                "five six seven",
            ],
        )

    def test_invalid_chunk_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text("content", max_words=10, overlap_words=10)


class RetrievalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.knowledge_dir = Path(self.temporary_directory.name)
        (self.knowledge_dir / "graphql.txt").write_text(
            "GraphQL APIs support queries, mutations, and subscriptions.",
            encoding="utf-8",
        )
        (self.knowledge_dir / "step-functions.txt").write_text(
            "AWS Step Functions orchestrates workflows and can send SQS messages.",
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_filename_metadata_guides_topic_classification(self) -> None:
        chunks = read_knowledge_chunks(self.knowledge_dir)
        topics_by_source = {
            chunk.source: chunk.topic
            for chunk in chunks
        }

        self.assertEqual(topics_by_source["graphql.txt"], "graphql")
        self.assertEqual(
            topics_by_source["step-functions.txt"],
            "orchestration",
        )

    def test_current_query_outweighs_previous_context(self) -> None:
        results = search(
            query="How does the workflow send SQS messages?",
            context="GraphQL subscriptions mutations GraphQL",
            knowledge_dir=self.knowledge_dir,
        )

        self.assertEqual(results[0].chunk.source, "step-functions.txt")
        self.assertIn("sqs", results[0].matched_terms)

    def test_formatted_result_exposes_retrieval_metadata(self) -> None:
        answer = search_best_match(
            "Step Functions workflow",
            knowledge_dir=self.knowledge_dir,
        )

        self.assertIn("Source: step-functions.txt", answer)
        self.assertIn("Topic: orchestration", answer)
        self.assertIn("Chunk: 1", answer)
        self.assertIn("Score:", answer)

    def test_empty_directory_returns_no_match(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            answer = search_best_match(
                "AWS",
                knowledge_dir=Path(directory),
            )

        self.assertEqual(
            answer,
            "I could not find relevant local knowledge for that topic.",
        )

    def test_stop_words_do_not_create_false_matches(self) -> None:
        results = search(
            query="What is a dead letter queue?",
            knowledge_dir=self.knowledge_dir,
        )

        self.assertEqual(results, [])

    def test_context_cannot_create_a_match_for_an_unrelated_query(self) -> None:
        results = search(
            query="Explain idempotency.",
            context="GraphQL subscriptions and Step Functions workflows",
            knowledge_dir=self.knowledge_dir,
        )

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main()
