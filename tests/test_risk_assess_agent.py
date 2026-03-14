import unittest

from app.agents._0_contracts import RiskAssessInput
from app.agents._eval_4_1_risk_assessor import RiskAssessAgent


class FakeGemma:
    def __init__(self, text: str):
        self._text = text

    def generate_text(self, contents, max_output_tokens=2000):
        return self._text


class RiskAssessAgentTest(unittest.TestCase):
    def test_filters_and_clamps_fields(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu": "Katsu Don",
                  "risk": 130,
                  "confidence": -0.2,
                  "suspected_ingredients": ["egg", "pork", "onion", "extra"],
                  "matched_avoid": ["egg", "random"],
                  "why_ko": "계란 가능성"
                },
                {
                  "menu": "NotInInput",
                  "risk": 10,
                  "confidence": 0.9,
                  "suspected_ingredients": ["x"],
                  "matched_avoid": ["x"],
                  "why_ko": "무시"
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Katsu Don"], avoid=["egg", "milk"])

        result = agent.run(req)

        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        self.assertEqual(item.menu, "Katsu Don")
        self.assertEqual(item.risk, 35)
        self.assertEqual(item.confidence, 0.0)
        self.assertEqual(item.suspected_ingredients, ["egg", "pork", "onion"])
        self.assertEqual(item.matched_avoid, [])
        self.assertEqual(item.avoid_evidence, [])
        self.assertEqual(item.why_ko, "기피 재료 근거 부족")

    def test_returns_empty_if_items_empty(self):
        fake = FakeGemma('{"items": []}')
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=[], avoid=["egg"])

        result = agent.run(req)

        self.assertEqual(result.items, [])

    def test_low_confidence_uncertain_becomes_no_match(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu": "Katsu Don",
                  "risk": 90,
                  "confidence": 0.2,
                  "suspected_ingredients": ["egg"],
                  "matched_avoid": ["egg"],
                  "avoid_evidence": [
                    {"ingredient": "egg", "evidence_type": "uncertain", "note_ko": "불확실"}
                  ],
                  "why_ko": "계란 같음"
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Katsu Don"], avoid=["egg"])

        result = agent.run(req)

        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        self.assertEqual(item.matched_avoid, [])
        self.assertEqual(item.avoid_evidence, [])
        self.assertLessEqual(item.risk, 35)

    def test_fills_missing_menu_items_with_conservative_fallback(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu": "Menu A",
                  "risk": 20,
                  "confidence": 0.9,
                  "suspected_ingredients": [],
                  "matched_avoid": [],
                  "avoid_evidence": [],
                  "why_ko": "기피 재료 근거 부족"
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Menu A", "Menu B"], avoid=["egg"])

        result = agent.run(req)

        self.assertEqual([it.menu for it in result.items], ["Menu A", "Menu B"])
        self.assertEqual(result.items[1].risk, 60)
        self.assertEqual(result.items[1].confidence, 0.0)
        self.assertEqual(result.items[1].why_ko, "위험도 판단 실패(항목 누락 보수적 처리)")

    def test_normalizes_avoid_matching_for_case_difference(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu": "Menu A",
                  "risk": 60,
                  "confidence": 0.9,
                  "suspected_ingredients": ["egg"],
                  "matched_avoid": ["Egg"],
                  "avoid_evidence": [
                    {"ingredient": "Egg", "evidence_type": "direct", "note_ko": "표기에 대소문자 차이"}
                  ],
                  "why_ko": "계란 근거"
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Menu A"], avoid=["egg"])

        result = agent.run(req)

        self.assertEqual(result.items[0].matched_avoid, ["egg"])
        self.assertEqual(result.items[0].avoid_evidence[0].ingredient, "egg")


if __name__ == "__main__":
    unittest.main()
