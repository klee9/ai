import unittest

from app.agents._0_contracts import AvoidEvidence, RiskItem, ScorePolicyInput
from app.agents._eval_4_2_score_policy import ScorePolicyAgent


class ScorePolicyAgentTest(unittest.TestCase):
    def test_score_calculation_and_sorting(self):
        agent = ScorePolicyAgent()
        req = ScorePolicyInput(
            risk_items=[
                RiskItem(
                    menu="A",
                    risk=20,
                    confidence=1.0,
                    suspected_ingredients=["x"],
                    matched_avoid=[],
                    avoid_evidence=[],
                    why_ko="안전",
                ),
                RiskItem(
                    menu="B",
                    risk=10,
                    confidence=0.5,
                    suspected_ingredients=["y"],
                    matched_avoid=["egg"],
                    avoid_evidence=[
                        AvoidEvidence(
                            ingredient="egg",
                            evidence_type="direct",
                            note_ko="계란 사용 가능성 높음",
                        )
                    ],
                    why_ko="주의",
                ),
            ],
            uncertainty_penalty=40,
        )

        result = agent.run(req)

        self.assertEqual(len(result.items), 2)
        # A: no-evidence risk=max(uncertainty 45, bounded prior 20)=45 -> 55
        # B: risk=max(structured 45, prior 10)=45 -> 100-45-20=35
        self.assertEqual(result.items[0].menu, "A")
        self.assertEqual(result.items[0].menu_original, "A")
        self.assertEqual(result.items[0].score, 55)
        self.assertEqual(result.items[1].menu, "B")
        self.assertEqual(result.items[1].menu_original, "B")
        self.assertEqual(result.items[1].score, 35)
        self.assertEqual(result.items[1].reason, "Caution: egg (direct evidence)")
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.menu, "A")

    def test_reason_language_en(self):
        agent = ScorePolicyAgent()
        req = ScorePolicyInput(
            risk_items=[
                RiskItem(
                    menu="B",
                    risk=10,
                    confidence=1.0,
                    suspected_ingredients=[],
                    matched_avoid=["egg"],
                    avoid_evidence=[
                        AvoidEvidence(ingredient="egg", evidence_type="direct", note_ko="")
                    ],
                    why_ko="",
                )
            ],
            uncertainty_penalty=40,
            lang="en",
        )
        result = agent.run(req)
        self.assertIn("Caution", result.items[0].reason)

    def test_no_evidence_uses_conservative_risk(self):
        agent = ScorePolicyAgent()
        req = ScorePolicyInput(
            risk_items=[
                RiskItem(
                    menu="C",
                    risk=90,
                    confidence=1.0,
                    suspected_ingredients=[],
                    matched_avoid=[],
                    avoid_evidence=[],
                    why_ko="기피 재료 근거 부족",
                )
            ],
            uncertainty_penalty=40,
        )
        result = agent.run(req)
        self.assertEqual(result.items[0].risk, 70)


if __name__ == "__main__":
    unittest.main()
