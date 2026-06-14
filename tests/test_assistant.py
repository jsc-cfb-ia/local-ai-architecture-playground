import unittest

from app.assistant import ArchitectureAssistant
from app.models import ModelError


class RecordingModel:
    def __init__(
        self,
        response: str = "Generated architecture answer.",
        responses: list[str] | None = None,
    ) -> None:
        self.response = response
        self.responses = list(responses or [])
        self.system_prompt = ""
        self.user_prompt = ""
        self.call_count = 0

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.call_count += 1
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        return self.responses.pop(0) if self.responses else self.response


class FailingModel:
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise ModelError("Local model unavailable.")


class ArchitectureAssistantTests(unittest.TestCase):
    def test_model_receives_question_and_retrieved_sources(self) -> None:
        model = RecordingModel()
        assistant = ArchitectureAssistant(model=model)

        response = assistant.answer(
            "How do Step Functions coordinate SQS?",
            context="We are designing a workflow.",
        )

        self.assertTrue(response.generated_by_model)
        self.assertEqual(response.answer_mode, "grounded")
        self.assertEqual(response.content, "Generated architecture answer.")
        self.assertIn("step-functions.txt", response.sources)
        self.assertIn("How do Step Functions", model.user_prompt)
        self.assertIn("[Source:", model.user_prompt)
        self.assertIn("designing a workflow", model.user_prompt)

    def test_model_failure_falls_back_to_lexical_retrieval(self) -> None:
        assistant = ArchitectureAssistant(model=FailingModel())

        response = assistant.answer("What is GraphQL?")

        self.assertFalse(response.generated_by_model)
        self.assertEqual(response.answer_mode, "retrieval_fallback")
        self.assertIn("Source: graphql.txt", response.content)
        self.assertEqual(response.warning, "Local model unavailable.")

    def test_no_model_uses_retrieval_only(self) -> None:
        assistant = ArchitectureAssistant()

        response = assistant.answer("What is AWS?")

        self.assertFalse(response.generated_by_model)
        self.assertEqual(response.answer_mode, "retrieval")
        self.assertIn("Source: aws.txt", response.content)

    def test_no_local_match_uses_general_model_knowledge(self) -> None:
        model = RecordingModel(
            "Consistent hashing distributes keys across changing nodes."
        )
        assistant = ArchitectureAssistant(model=model)

        response = assistant.answer("Explain consistent hashing.")

        self.assertTrue(response.generated_by_model)
        self.assertEqual(response.answer_mode, "general")
        self.assertEqual(response.sources, ())
        self.assertIn("No relevant local source", response.warning)
        self.assertNotIn("Local knowledge:", model.user_prompt)

    def test_general_knowledge_can_be_disabled(self) -> None:
        assistant = ArchitectureAssistant(
            model=RecordingModel(),
            allow_general_knowledge=False,
        )

        response = assistant.answer("Explain consistent hashing.")

        self.assertFalse(response.generated_by_model)
        self.assertEqual(response.answer_mode, "no_match")

    def test_insufficient_retrieved_context_uses_general_knowledge(self) -> None:
        model = RecordingModel(
            responses=[
                "INSUFFICIENT_CONTEXT",
                "Idempotency prevents duplicate side effects.",
            ]
        )
        assistant = ArchitectureAssistant(model=model)

        response = assistant.answer(
            "Explain idempotency in event-driven systems."
        )

        self.assertEqual(response.answer_mode, "general")
        self.assertEqual(response.sources, ())
        self.assertIn(
            "retrieved local chunks were insufficient",
            response.warning,
        )

    def test_partial_local_match_skips_grounded_model_call(self) -> None:
        model = RecordingModel(
            "AWS Bash is not an official AWS product."
        )
        assistant = ArchitectureAssistant(model=model)

        response = assistant.answer("que es aws bash")

        self.assertEqual(response.answer_mode, "general")
        self.assertEqual(response.sources, ())
        self.assertEqual(model.call_count, 1)
        self.assertNotIn("Local knowledge:", model.user_prompt)
        self.assertIn("matched only part", response.warning)


if __name__ == "__main__":
    unittest.main()
