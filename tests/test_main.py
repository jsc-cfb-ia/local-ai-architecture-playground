import os
import unittest
from unittest.mock import patch

from app.main import get_runtime_status, should_save_to_memory


class RuntimeStatusTests(unittest.TestCase):
    def test_default_runtime_uses_ollama(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            status = get_runtime_status()

        self.assertIn("Provider: ollama", status)
        self.assertIn("Model: qwen3:4b", status)

    def test_ollama_runtime_uses_qwen3_4b_by_default(self) -> None:
        with patch.dict(
            os.environ,
            {"LOCAL_AI_PROVIDER": "ollama"},
            clear=True,
        ):
            status = get_runtime_status()

        self.assertIn("Provider: ollama", status)
        self.assertIn("Model: qwen3:4b", status)
        self.assertIn("No local evidence: general model knowledge", status)
        self.assertIn("lexical retrieval fallback", status)

    def test_retrieval_runtime_can_be_selected_explicitly(self) -> None:
        with patch.dict(
            os.environ,
            {"LOCAL_AI_PROVIDER": "retrieval"},
            clear=True,
        ):
            status = get_runtime_status()

        self.assertIn("Provider: retrieval", status)
        self.assertIn("Generation: disabled", status)

    def test_status_command_is_not_saved_to_memory(self) -> None:
        self.assertFalse(should_save_to_memory("status"))


if __name__ == "__main__":
    unittest.main()
