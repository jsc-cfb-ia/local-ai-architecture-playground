from dataclasses import dataclass

from app.models import LanguageModel, ModelError
from app.retriever import SearchResult, search, search_best_match

SYSTEM_PROMPT = """You are a local architecture engineering assistant.
Answer using only the supplied local knowledge.
If the local knowledge cannot answer the question, respond with exactly:
INSUFFICIENT_CONTEXT
Do not invent AWS features, guarantees, or implementation details.
Keep the answer concise and practical."""

GENERAL_SYSTEM_PROMPT = """You are a local architecture engineering assistant.
Answer from your built-in general knowledge because no relevant document was
found in the local knowledge base.
Be concise and practical.
Do not claim that the answer came from local documents.
State uncertainty when details may depend on current product documentation."""


@dataclass(frozen=True)
class AssistantResponse:
    """Structured response suitable for a CLI, skill, API, or agent tool."""

    content: str
    sources: tuple[str, ...]
    generated_by_model: bool
    answer_mode: str
    warning: str = ""

    def format_for_terminal(self) -> str:
        sections = [self.content]

        if self.sources:
            sections.append(f"Sources: {', '.join(self.sources)}")

        sections.append(f"Answer mode: {self.answer_mode}")

        if self.warning:
            sections.append(f"Note: {self.warning}")

        return "\n\n".join(sections)


class ArchitectureAssistant:
    """Orchestrate retrieval and optional grounded response generation."""

    def __init__(
        self,
        model: LanguageModel | None = None,
        retrieval_limit: int = 2,
        allow_general_knowledge: bool = True,
    ) -> None:
        self.model = model
        self.retrieval_limit = retrieval_limit
        self.allow_general_knowledge = allow_general_knowledge

    def answer(self, question: str, context: str = "") -> AssistantResponse:
        results = search(
            query=question,
            context=context,
            limit=self.retrieval_limit,
        )

        if not results:
            if self.model is not None and self.allow_general_knowledge:
                return self._answer_from_general_knowledge(
                    question=question,
                    context=context,
                )

            return AssistantResponse(
                content=(
                    "I could not find relevant local knowledge for that topic."
                ),
                sources=(),
                generated_by_model=False,
                answer_mode="no_match",
            )

        sources = tuple(
            dict.fromkeys(result.chunk.source for result in results)
        )

        if self.model is None:
            return AssistantResponse(
                content=search_best_match(question, context=context),
                sources=sources[:1],
                generated_by_model=False,
                answer_mode="retrieval",
            )

        try:
            content = self.model.generate(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=self._build_grounded_prompt(
                    question=question,
                    context=context,
                    results=results,
                ),
            )
        except ModelError as error:
            return AssistantResponse(
                content=search_best_match(question, context=context),
                sources=sources[:1],
                generated_by_model=False,
                answer_mode="retrieval_fallback",
                warning=str(error),
            )

        if self._is_insufficient_context(content):
            if not self.allow_general_knowledge:
                return AssistantResponse(
                    content=(
                        "The local knowledge is insufficient to answer "
                        "that question."
                    ),
                    sources=(),
                    generated_by_model=False,
                    answer_mode="no_match",
                )

            return self._answer_from_general_knowledge(
                question=question,
                context=context,
                warning=(
                    "The retrieved local chunks were insufficient; this "
                    "answer uses the model's general knowledge."
                ),
            )

        return AssistantResponse(
            content=content,
            sources=sources,
            generated_by_model=True,
            answer_mode="grounded",
        )

    def _answer_from_general_knowledge(
        self,
        question: str,
        context: str,
        warning: str = (
            "No relevant local source was found; this answer uses the "
            "model's general knowledge."
        ),
    ) -> AssistantResponse:
        try:
            content = self.model.generate(
                system_prompt=GENERAL_SYSTEM_PROMPT,
                user_prompt=(
                    f"Recent conversation context:\n"
                    f"{context or '(none)'}\n\n"
                    f"Question:\n{question}"
                ),
            )
        except ModelError as error:
            return AssistantResponse(
                content=(
                    "I could not find relevant local knowledge for that topic."
                ),
                sources=(),
                generated_by_model=False,
                answer_mode="no_match",
                warning=str(error),
            )

        return AssistantResponse(
            content=content,
            sources=(),
            generated_by_model=True,
            answer_mode="general",
            warning=warning,
        )

    @staticmethod
    def _is_insufficient_context(content: str) -> bool:
        normalized = content.strip().lower()
        signals = (
            "insufficient_context",
            "cannot answer this question based on",
            "does not include information",
            "not enough information",
        )

        return any(signal in normalized for signal in signals)

    @staticmethod
    def _build_grounded_prompt(
        question: str,
        context: str,
        results: list[SearchResult],
    ) -> str:
        knowledge = "\n\n".join(
            (
                f"[Source: {result.chunk.source}; "
                f"Topic: {result.chunk.topic}; "
                f"Chunk: {result.chunk.chunk_index}]\n"
                f"{result.chunk.content}"
            )
            for result in results
        )

        return (
            f"Recent conversation context:\n{context or '(none)'}\n\n"
            f"Local knowledge:\n{knowledge}\n\n"
            f"Question:\n{question}"
        )
