import json
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class ModelError(RuntimeError):
    """Raised when a local language model cannot generate a response."""


class LanguageModel(Protocol):
    """Minimal model contract reusable by assistants, tools, and agents."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Generate text from system and user instructions."""


@dataclass(frozen=True)
class OllamaModel:
    """Language model adapter for Ollama's local HTTP API."""

    model: str = "qwen3:4b"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    think: bool = False
    keep_alive: str = "5m"
    opener: Callable[..., object] = urlopen

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        if not self.think:
            system_prompt = f"{system_prompt}\n/no_think"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            "think": self.think,
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0.2,
            },
        }
        request = Request(
            url=f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with self.opener(
                request,
                timeout=self.timeout_seconds,
            ) as response:
                body = json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise ModelError(
                "Ollama is unavailable. Start Ollama and verify the model "
                f"'{self.model}' is installed."
            ) from error
        except json.JSONDecodeError as error:
            raise ModelError("Ollama returned invalid JSON.") from error

        try:
            content = body["message"]["content"].strip()
        except (KeyError, AttributeError, TypeError) as error:
            raise ModelError("Ollama returned an unexpected response.") from error

        if not content:
            raise ModelError("Ollama returned an empty response.")

        return self._remove_thinking_content(content)

    @staticmethod
    def _remove_thinking_content(content: str) -> str:
        """Prevent model reasoning traces from leaking into tool output."""

        if "</think>" in content:
            content = content.rsplit("</think>", maxsplit=1)[1].strip()

        if not content:
            raise ModelError(
                "Ollama returned reasoning without a final answer."
            )

        return content
