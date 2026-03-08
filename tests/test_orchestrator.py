import unittest
from unittest.mock import patch

from app.agents.contracts import ScoredItem
from app.agents.orchestrator import MenuAgentOrchestrator


class FakeGemma:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def image_part_from_bytes(self, data, mime):
        return {"data": data, "mime": mime}

    def generate_text(self, contents, max_output_tokens=900):
        self.calls.append((contents, max_output_tokens))
        if not self._responses:
            raise RuntimeError("No fake response prepared")
        return self._responses.pop(0)


class OrchestratorTest(unittest.TestCase):
    @patch("app.agents.orchestrator.load_image_from_url")
    def test_runs_full_pipeline(self, mock_load_image):
        mock_load_image.return_value = (b"img", "image/png")
        fake = FakeGemma(
            responses=[
                '{"items":["Katsu Don"]}',  # ExtractAgent
                '{"items":[{"menu":"Katsu Don","risk":60,"confidence":1.0,"suspected_ingredients":["egg"],"matched_avoid":["egg"],"why_ko":"계란 포함 가능"}]}',  # RiskAssessAgent
                '{"items":[{"original":"Caution: egg (inference)","translated":"egg 추정으로 주의","confidence":1.0}]}'  # Translate reason
            ]
        )
        orchestrator = MenuAgentOrchestrator(fake, uncertainty_penalty=40)

        result = orchestrator.run("https://example.com/menu.png", ["egg"])

        self.assertEqual(result.items_extracted, ["Katsu Don"])
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].menu, "Katsu Don")
        self.assertEqual(result.items[0].score, 40)
        self.assertEqual(result.items[0].reason, "egg 추정으로 주의")
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.menu, "Katsu Don")
        self.assertIn("image_load", result.timings_ms)
        self.assertIn("preprocess", result.timings_ms)
        self.assertIn("extract", result.timings_ms)
        self.assertIn("risk_assess", result.timings_ms)
        self.assertIn("score_policy", result.timings_ms)
        self.assertIn("total", result.timings_ms)

    @patch("app.agents.orchestrator.load_image_from_url")
    def test_fallback_score_when_risk_assess_fails(self, mock_load_image):
        mock_load_image.return_value = (b"img", "image/png")
        fake = FakeGemma(
            responses=[
                '{"items":["Menu A","Menu B"]}',  # ExtractAgent
                "NOT JSON",  # RiskAssess 1st try
                "ALSO NOT JSON",  # RiskAssess retry
                '{"items":[{"original":"Risk assessment failed (conservative fallback)","translated":"위험도 판단 실패(보수적 처리)","confidence":1.0}]}'  # Translate reason
            ]
        )
        orchestrator = MenuAgentOrchestrator(fake, uncertainty_penalty=40, max_risk_retries=1)

        result = orchestrator.run("https://example.com/menu.png", ["egg"])

        self.assertEqual(result.items_extracted, ["Menu A", "Menu B"])
        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].score, 0)
        self.assertEqual(result.items[0].risk, 100)
        self.assertEqual(result.items[0].confidence, 0.0)
        self.assertEqual(result.items[0].reason, "위험도 판단 실패(보수적 처리)")
        self.assertIn("risk_assess", result.timings_ms)
        self.assertIn("score_policy", result.timings_ms)

    def test_localize_reasons_uses_single_batch_translate(self):
        fake = FakeGemma(
            responses=[
                '{"items":[{"original":"Caution: egg (direct evidence)","translated":"계란 근거로 주의","confidence":1.0}]}'
            ]
        )
        orchestrator = MenuAgentOrchestrator(fake, uncertainty_penalty=40)

        items = [
            ScoredItem(
                menu="A",
                menu_original="A",
                score=10,
                risk=90,
                confidence=0.9,
                matched_avoid=["egg"],
                suspected_ingredients=["egg"],
                reason="Caution: egg (direct evidence)",
            ),
            ScoredItem(
                menu="B",
                menu_original="B",
                score=12,
                risk=88,
                confidence=0.9,
                matched_avoid=["egg"],
                suspected_ingredients=["egg"],
                reason="Caution: egg (direct evidence)",
            ),
        ]

        orchestrator._localize_item_reasons(items, "ko")

        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(items[0].reason, "계란 근거로 주의")
        self.assertEqual(items[1].reason, "계란 근거로 주의")


if __name__ == "__main__":
    unittest.main()
