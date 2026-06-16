import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from app.embeddings import OllamaEmbeddingModel
from app.models import ModelError
from app.retriever import PROJECT_ROOT, meaningful_tokens, search
from app.semantic_retriever import search_semantic

DEFAULT_DATASET = PROJECT_ROOT / "evaluation" / "questions.json"


@dataclass(frozen=True)
class EvaluationCase:
    """One retrieval question with acceptable top-ranked sources."""

    case_id: str
    question: str
    expected_sources: tuple[str, ...]
    expected_terms: tuple[str, ...]


@dataclass(frozen=True)
class EvaluationResult:
    """Observed top-ranked source for one evaluation case."""

    case: EvaluationCase
    actual_source: str
    score: float
    missing_terms: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return (
            self.actual_source in self.case.expected_sources
            and not self.missing_terms
        )


def load_cases(dataset_path: Path = DEFAULT_DATASET) -> list[EvaluationCase]:
    """Load retrieval evaluation cases from JSON."""

    raw_cases = json.loads(dataset_path.read_text(encoding="utf-8"))

    return [
        EvaluationCase(
            case_id=raw_case["id"],
            question=raw_case["question"],
            expected_sources=tuple(raw_case["expected_sources"]),
            expected_terms=tuple(raw_case["expected_terms"]),
        )
        for raw_case in raw_cases
    ]


def evaluate_cases(
    cases: list[EvaluationCase],
    strategy: str = "lexical",
) -> list[EvaluationResult]:
    """Evaluate whether retrieval ranks an expected source first."""

    results = []
    embedding_model = (
        OllamaEmbeddingModel()
        if strategy == "semantic"
        else None
    )

    for case in cases:
        matches = (
            search_semantic(
                case.question,
                embedding_model=embedding_model,
                limit=1,
            )
            if strategy == "semantic"
            else search(case.question, limit=1)
        )
        top_match = matches[0] if matches else None
        chunk_terms = (
            set(meaningful_tokens(top_match.chunk.content))
            if top_match
            else set()
        )
        results.append(
            EvaluationResult(
                case=case,
                actual_source=(
                    top_match.chunk.source if top_match else "no_match"
                ),
                score=top_match.score if top_match else 0.0,
                missing_terms=tuple(
                    term
                    for term in case.expected_terms
                    if term not in chunk_terms
                ),
            )
        )

    return results


def format_report(
    results: list[EvaluationResult],
    strategy: str = "lexical",
) -> str:
    """Format a human-readable retrieval evaluation report."""

    lines = [f"=== Retrieval Evaluation ({strategy}) ==="]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        expected = ", ".join(result.case.expected_sources)
        missing_terms = ", ".join(result.missing_terms) or "none"
        lines.append(
            f"{status} {result.case.case_id}\n"
            f"  question: {result.case.question}\n"
            f"  expected: {expected}\n"
            f"  actual: {result.actual_source} "
            f"(score={result.score:.2f})\n"
            f"  missing expected terms: {missing_terms}"
        )

    passed = sum(result.passed for result in results)
    total = len(results)
    accuracy = passed / total if total else 0.0
    lines.append(
        f"Summary: {passed}/{total} passed "
        f"({accuracy:.0%} top-1 source accuracy)"
    )

    return "\n\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate top-ranked local retrieval sources."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help="Path to a JSON evaluation dataset.",
    )
    parser.add_argument(
        "--strategy",
        choices=("lexical", "semantic", "both"),
        default="lexical",
        help="Retrieval strategy to evaluate.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cases = load_cases(args.dataset)
    strategies = (
        ("lexical", "semantic")
        if args.strategy == "both"
        else (args.strategy,)
    )
    all_results = []

    for strategy in strategies:
        try:
            results = evaluate_cases(cases, strategy=strategy)
        except ModelError as error:
            print(f"Semantic evaluation unavailable: {error}")
            raise SystemExit(1) from error

        all_results.extend(results)
        print(format_report(results, strategy=strategy))

    if not all(result.passed for result in all_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
