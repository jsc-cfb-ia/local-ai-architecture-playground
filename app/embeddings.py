import json
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from app.models import ModelError


class EmbeddingModel(Protocol):
    """Minimal contract for local text embedding models."""

    model: str

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate one embedding vector per input text."""


@dataclass(frozen=True)
class OllamaEmbeddingModel:
    """Embedding adapter for Ollama's local /api/embed endpoint."""

    model: str = "nomic-embed-text"
    base_url: str = "http://localhost:11434"
    timeout_seconds: float = 120.0
    keep_alive: str = "5m"
    opener: Callable[..., object] = urlopen

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        payload = {
            "model": self.model,
            "input": texts,
            "keep_alive": self.keep_alive,
        }
        request = Request(
            url=f"{self.base_url.rstrip('/')}/api/embed",
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
                "Ollama embeddings are unavailable. Start Ollama and verify "
                f"the model '{self.model}' is installed."
            ) from error
        except json.JSONDecodeError as error:
            raise ModelError("Ollama returned invalid embedding JSON.") from error

        embeddings = body.get("embeddings")

        if not isinstance(embeddings, list):
            raise ModelError("Ollama returned an unexpected embedding response.")

        if len(embeddings) != len(texts):
            raise ModelError("Ollama returned the wrong number of embeddings.")

        return [
            [float(value) for value in embedding]
            for embedding in embeddings
        ]
