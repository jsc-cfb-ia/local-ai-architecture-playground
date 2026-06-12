from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONTEXT_FILE = PROJECT_ROOT / "memory" / "context.txt"


def save_context(question: str) -> None:
    """Save the latest user question as conversation context."""

    CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)

    with CONTEXT_FILE.open("a", encoding="utf-8") as file:
        file.write(f"{question}\n")


def read_context(max_lines: int = 3) -> str:
    """Read the latest conversation context."""

    if not CONTEXT_FILE.exists():
        return ""

    lines = CONTEXT_FILE.read_text(
        encoding="utf-8"
    ).splitlines()

    recent_lines = lines[-max_lines:]

    return " ".join(recent_lines).strip()


def clear_context() -> None:
    """Clear conversation context."""

    CONTEXT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CONTEXT_FILE.write_text("", encoding="utf-8")
