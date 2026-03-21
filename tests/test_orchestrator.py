import unittest
from types import SimpleNamespace
from unittest.mock import patch

from app.agents._0_contracts import ScoredItem
from app.agents._0_orchestrator import MenuAgentOrchestrator


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
    @patch("app.agents._0_orchestrator.clean_menu_candidates")
    @patch("app.agents._0_orchestrator.OCRAgent")
    @patch("app.agents._0_orchestrator.load_image")
    def test_runs_full_pipeline(self, mock_load_image, mock_ocr_cls, mock_clean_menu_candidates):
        mock_load_image.return_value = (b"img", "image/png")
        mock_ocr_cls.return_value.run.return_value = SimpleNamespace(
            lines=["Egg Burger"],
            resolved_lang="en",
            lang_detection_source="manual",
        )
        mock_clean_menu_candidates.return_value = SimpleNamespace(cleaned_items=["Egg Burger"])
        fake = FakeGemma(
            responses=[
                '{"items":["Egg Burger"]}',  # ExtractAgent
                '{"items":[{"menu_name":"Egg Burger","confidence":1.0,"suspects":[{"canonical":"egg","evidence_type":"direct","evidence_text":"Egg","reason":"menu directly names egg","confidence":0.95}]}]}',  # RiskAssessAgent
                '{"items":[{"original":"Caution: egg (inference)","translated":"egg 추정으로 주의","confidence":1.0}]}'  # Translate reason
            ]
        )
        orchestrator = MenuAgentOrchestrator(fake, uncertainty_penalty=40)
        orchestrator.extract_agent.run_lines_with_image = lambda **kwargs: SimpleNamespace(
            menu_texts=["Egg Burger"],
            to_extract_output=lambda: SimpleNamespace(items=["Egg Burger"]),
        )
        orchestrator.translate_only = lambda texts, source_lang="auto", target_lang="en": SimpleNamespace(
            items=[
                SimpleNamespace(
                    translated="egg 추정으로 주의" if str(text).startswith("Caution:") else str(text)
                )
                for text in texts
            ]
        )
        orchestrator.bbox_agent.run = lambda **kwargs: ("", [], "", 0)

        result = orchestrator.run("https://example.com/menu.png", ["egg"])

        self.assertEqual(result.items_extracted, ["Egg Burger"])
        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].menu, "Egg Burger")
        self.assertGreaterEqual(result.items[0].risk, 50)
        self.assertLessEqual(result.items[0].score, 50)
        self.assertEqual(result.items[0].reason, "egg 추정으로 주의")
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.menu, "Egg Burger")
        self.assertIn("image_load", result.timings_ms)
        self.assertIn("preprocess", result.timings_ms)
        self.assertIn("extract", result.timings_ms)
        self.assertIn("risk_assess", result.timings_ms)
        self.assertIn("score_policy", result.timings_ms)
        self.assertIn("total", result.timings_ms)

    @patch("app.agents._0_orchestrator.clean_menu_candidates")
    @patch("app.agents._0_orchestrator.OCRAgent")
    @patch("app.agents._0_orchestrator.load_image")
    def test_fallback_score_when_risk_assess_fails(self, mock_load_image, mock_ocr_cls, mock_clean_menu_candidates):
        mock_load_image.return_value = (b"img", "image/png")
        mock_ocr_cls.return_value.run.return_value = SimpleNamespace(
            lines=["Menu A", "Menu B"],
            resolved_lang="en",
            lang_detection_source="manual",
        )
        mock_clean_menu_candidates.return_value = SimpleNamespace(cleaned_items=["Menu A", "Menu B"])
        fake = FakeGemma(
            responses=[
                '{"items":["Menu A","Menu B"]}',  # ExtractAgent
                "NOT JSON",  # RiskAssess 1st try
                "ALSO NOT JSON",  # RiskAssess retry
                '{"items":[{"original":"Risk assessment failed (conservative fallback)","translated":"위험도 판단 실패(보수적 처리)","confidence":1.0}]}'  # Unused for no-match fallback reason
            ]
        )
        orchestrator = MenuAgentOrchestrator(fake, uncertainty_penalty=40, max_risk_retries=1)
        orchestrator.extract_agent.run_lines_with_image = lambda **kwargs: SimpleNamespace(
            menu_texts=["Menu A", "Menu B"],
            to_extract_output=lambda: SimpleNamespace(items=["Menu A", "Menu B"]),
        )
        orchestrator.bbox_agent.run = lambda **kwargs: ("", [], "", 0)

        result = orchestrator.run("https://example.com/menu.png", ["egg"])

        self.assertEqual(result.items_extracted, ["Menu A", "Menu B"])
        self.assertEqual(len(result.items), 2)
        self.assertGreaterEqual(result.items[0].risk, 50)
        self.assertLessEqual(result.items[0].score, 50)
        self.assertEqual(result.items[0].confidence, 0.0)
        self.assertEqual(result.items[0].reason, "뚜렷한 기피 재료는 확인되지 않았어요. 다만 정보가 부족해 점수는 보수적으로 계산했어요.")
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
