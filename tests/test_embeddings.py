import json
import unittest
from urllib.error import URLError

from app.embeddings import OllamaEmbeddingModel
from app.models import ModelError


class FakeResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


class OllamaEmbeddingModelTests(unittest.TestCase):
    def test_embed_calls_ollama_embed_endpoint(self) -> None:
        captured = {}

        def opener(request: object, timeout: float) -> FakeResponse:
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(
                {"embeddings": [[1.0, 0.0], [0.0, 1.0]]}
            )

        model = OllamaEmbeddingModel(
            model="embed-model",
            opener=opener,
            timeout_seconds=10,
        )

        embeddings = model.embed(["alpha", "beta"])

        self.assertEqual(embeddings, [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(
            captured["url"],
            "http://localhost:11434/api/embed",
        )
        self.assertEqual(captured["payload"]["model"], "embed-model")
        self.assertEqual(captured["payload"]["input"], ["alpha", "beta"])
        self.assertEqual(captured["timeout"], 10)

    def test_connection_error_becomes_model_error(self) -> None:
        def opener(request: object, timeout: float) -> object:
            raise URLError("connection refused")

        model = OllamaEmbeddingModel(opener=opener)

        with self.assertRaisesRegex(ModelError, "embeddings are unavailable"):
            model.embed(["question"])


if __name__ == "__main__":
    unittest.main()
