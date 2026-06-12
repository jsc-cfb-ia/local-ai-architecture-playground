import json
import unittest
from urllib.error import URLError

from app.models import ModelError, OllamaModel


class FakeResponse:
    def __init__(self, body: dict[str, object]) -> None:
        self.body = body

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.body).encode("utf-8")


class OllamaModelTests(unittest.TestCase):
    def test_generate_calls_local_chat_endpoint(self) -> None:
        captured = {}

        def opener(request: object, timeout: float) -> FakeResponse:
            captured["url"] = request.full_url
            captured["payload"] = json.loads(request.data.decode("utf-8"))
            captured["timeout"] = timeout
            return FakeResponse(
                {"message": {"content": " Grounded response. "}}
            )

        model = OllamaModel(
            model="small-model",
            opener=opener,
            timeout_seconds=10,
        )

        result = model.generate("System", "Question")

        self.assertEqual(result, "Grounded response.")
        self.assertEqual(
            captured["url"],
            "http://localhost:11434/api/chat",
        )
        self.assertEqual(captured["payload"]["model"], "small-model")
        self.assertTrue(
            captured["payload"]["messages"][0]["content"].endswith(
                "/no_think"
            )
        )
        self.assertFalse(captured["payload"]["stream"])
        self.assertFalse(captured["payload"]["think"])
        self.assertEqual(captured["payload"]["keep_alive"], "5m")
        self.assertEqual(captured["timeout"], 10)

    def test_connection_error_becomes_model_error(self) -> None:
        def opener(request: object, timeout: float) -> object:
            raise URLError("connection refused")

        model = OllamaModel(opener=opener)

        with self.assertRaisesRegex(ModelError, "Ollama is unavailable"):
            model.generate("System", "Question")

    def test_thinking_trace_is_removed_from_content(self) -> None:
        def opener(request: object, timeout: float) -> FakeResponse:
            return FakeResponse(
                {
                    "message": {
                        "content": (
                            "Private reasoning.</think>\nFinal answer."
                        )
                    }
                }
            )

        model = OllamaModel(opener=opener)

        self.assertEqual(
            model.generate("System", "Question"),
            "Final answer.",
        )


if __name__ == "__main__":
    unittest.main()
