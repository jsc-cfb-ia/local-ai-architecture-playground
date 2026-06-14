import os
import time
from pathlib import Path

from app.assistant import ArchitectureAssistant
from app.context_manager import clear_context, read_context, save_context
from app.models import OllamaModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MEMORY_FILE = PROJECT_ROOT / "memory" / "history.txt"
DEFAULT_PROVIDER = "ollama"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"


def create_assistant() -> ArchitectureAssistant:
    """Create the configured local assistant."""

    provider = os.getenv(
        "LOCAL_AI_PROVIDER",
        DEFAULT_PROVIDER,
    ).lower().strip()

    if provider == "ollama":
        return ArchitectureAssistant(
            model=OllamaModel(
                model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
                base_url=os.getenv(
                    "OLLAMA_BASE_URL",
                    "http://localhost:11434",
                ),
            )
        )

    return ArchitectureAssistant()


ASSISTANT = create_assistant()


def get_runtime_status() -> str:
    """Describe the configured local runtime."""

    provider = os.getenv(
        "LOCAL_AI_PROVIDER",
        DEFAULT_PROVIDER,
    ).lower().strip()

    if provider == "ollama":
        model = os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)
        base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        return (
            "Provider: ollama\n"
            f"Model: {model}\n"
            f"Endpoint: {base_url}\n"
            "Grounded answers: local knowledge + model\n"
            "No local evidence: general model knowledge\n"
            "Model unavailable: lexical retrieval fallback"
        )

    return (
        "Provider: retrieval\n"
        "Model: none\n"
        "Generation: disabled"
    )


def save_memory(question: str, answer: str) -> None:
    """Save useful conversation memory locally."""

    with MEMORY_FILE.open("a", encoding="utf-8") as file:
        file.write(f"User: {question}\n")
        file.write(f"Assistant: {answer}\n\n")


def read_memory() -> str:
    """Read conversation memory."""

    if not MEMORY_FILE.exists():
        return "No memory found yet."

    content = MEMORY_FILE.read_text(encoding="utf-8").strip()

    return content if content else "Memory is empty."


def get_response(question: str) -> str:
    """Return a response using context-aware local retrieval."""

    normalized_question = question.lower().strip()

    if normalized_question == "hello":
        return "Hello 👋"

    if normalized_question == "memory":
        return read_memory()

    if normalized_question == "clear context":
        clear_context()
        return "Context cleared."

    if normalized_question == "status":
        return get_runtime_status()

    context = read_context()

    return ASSISTANT.answer(question, context=context).format_for_terminal()


def should_save_to_memory(question: str) -> bool:
    """Decide whether the question should be stored."""

    command = question.lower().strip()

    internal_commands = {
        "memory",
        "status",
        "exit",
        "clear context",
    }

    return command not in internal_commands


def main() -> None:
    """Run the local AWS Architecture Assistant."""

    MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)

    print("Local AWS Architecture Assistant")
    print("Type 'exit' to finish.")
    print("Type 'memory' to view conversation memory.")
    print("Type 'status' to inspect the configured runtime.")
    print("Type 'clear context' to reset recent context.\n")

    while True:
        question = input("You: ").strip()

        if question.lower() == "exit":
            print("Goodbye 👋")
            break

        started_at = time.perf_counter()
        print("\nAssistant: processing...", flush=True)
        answer = get_response(question)
        elapsed_seconds = time.perf_counter() - started_at

        print(f"\nAssistant ({elapsed_seconds:.1f}s):\n{answer}\n")

        if should_save_to_memory(question):
            save_memory(question, answer)
            save_context(question)


if __name__ == "__main__":
    main()
