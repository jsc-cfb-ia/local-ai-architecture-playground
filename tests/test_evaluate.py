import unittest

from app.evaluate import evaluate_cases, format_report, load_cases


class RetrievalEvaluationTests(unittest.TestCase):
    def test_demo_7_dataset_contains_ten_cases(self) -> None:
        cases = load_cases()

        self.assertEqual(len(cases), 10)
        self.assertEqual(len({case.case_id for case in cases}), 10)

    def test_all_demo_7_expected_sources_rank_first(self) -> None:
        results = evaluate_cases(load_cases())

        self.assertTrue(all(result.passed for result in results))
        self.assertIn("10/10 passed", format_report(results))
        self.assertIn("lexical", format_report(results))


if __name__ == "__main__":
    unittest.main()
