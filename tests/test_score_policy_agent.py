import unittest

from app.agents._0_contracts import AvoidEvidence, RiskItem, ScorePolicyInput
from app.agents._eval_4_2_score_policy import ScorePolicyAgent


class ScorePolicyAgentTest(unittest.TestCase):
    def test_probability_scoring_and_sorting(self):
        agent = ScorePolicyAgent()
        req = ScorePolicyInput(
            risk_items=[
                RiskItem(
                    menu="A",
                    confidence=1.0,
                    suspected_ingredients=[],
                    matched_avoid=[],
                    avoid_evidence=[],
                ),
                RiskItem(
                    menu="B",
                    confidence=0.8,
                    suspected_ingredients=["egg"],
                    matched_avoid=["egg"],
                    avoid_evidence=[
                        AvoidEvidence(
                            ingredient="egg",
                            canonical="egg",
                            evidence_type="direct",
                            confidence=0.9,
                        )
                    ],
                ),
            ],
            uncertainty_penalty=40,
        )

        result = agent.run(req)

        self.assertEqual(len(result.items), 2)
        self.assertEqual(result.items[0].menu, "A")
        self.assertEqual(result.items[1].menu, "B")
        self.assertGreater(result.items[0].score, result.items[1].score)
        self.assertIn("Caution:", result.items[1].reason)
        self.assertIn("p=", result.items[1].reason)
        self.assertIsNotNone(result.best)
        self.assertEqual(result.best.menu, "A")

    def test_no_evidence_still_has_uncertainty_risk(self):
        agent = ScorePolicyAgent()
        req = ScorePolicyInput(
            risk_items=[
                RiskItem(
                    menu="Chef Special Set",
                    confidence=1.0,
                    suspected_ingredients=[],
                    matched_avoid=[],
                    avoid_evidence=[],
                )
            ],
            uncertainty_penalty=40,
        )

        result = agent.run(req)
        self.assertGreaterEqual(result.items[0].risk, 8)
        self.assertLessEqual(result.items[0].risk, 90)
        self.assertIn("Insufficient avoid-ingredient evidence", result.items[0].reason)

    def test_stronger_evidence_has_higher_risk(self):
        agent = ScorePolicyAgent()
        req = ScorePolicyInput(
            risk_items=[
                RiskItem(
                    menu="Weak Dish",
                    confidence=0.9,
                    matched_avoid=["egg"],
                    avoid_evidence=[
                        AvoidEvidence(
                            ingredient="egg",
                            canonical="egg",
                            evidence_type="weak_inference",
                            confidence=0.8,
                        )
                    ],
                ),
                RiskItem(
                    menu="Direct Dish",
                    confidence=0.9,
                    matched_avoid=["egg"],
                    avoid_evidence=[
                        AvoidEvidence(
                            ingredient="egg",
                            canonical="egg",
                            evidence_type="direct",
                            confidence=0.8,
                            evidence_text="egg",
                        )
                    ],
                ),
            ],
            uncertainty_penalty=40,
        )

        result = agent.run(req)
        by_menu = {item.menu: item for item in result.items}
        self.assertGreater(by_menu["Direct Dish"].risk, by_menu["Weak Dish"].risk)

    def test_multiple_ingredients_accumulate_risk(self):
        agent = ScorePolicyAgent()
        req = ScorePolicyInput(
            risk_items=[
                RiskItem(
                    menu="Single",
                    confidence=0.9,
                    matched_avoid=["egg"],
                    avoid_evidence=[
                        AvoidEvidence(
                            ingredient="egg",
                            canonical="egg",
                            evidence_type="alias",
                            confidence=0.7,
                        )
                    ],
                ),
                RiskItem(
                    menu="Multi",
                    confidence=0.9,
                    matched_avoid=["egg", "milk"],
                    avoid_evidence=[
                        AvoidEvidence(
                            ingredient="egg",
                            canonical="egg",
                            evidence_type="alias",
                            confidence=0.7,
                        ),
                        AvoidEvidence(
                            ingredient="milk",
                            canonical="milk",
                            evidence_type="alias",
                            confidence=0.7,
                        ),
                    ],
                ),
            ],
            uncertainty_penalty=40,
        )

        result = agent.run(req)
        by_menu = {item.menu: item for item in result.items}
        self.assertGreater(by_menu["Multi"].risk, by_menu["Single"].risk)

    def test_prefers_menu_original_for_menu_profile_and_output(self):
        agent = ScorePolicyAgent()
        req = ScorePolicyInput(
            risk_items=[
                RiskItem(
                    menu="스페인 초리소",
                    menu_original="Chorizo Espanol",
                    confidence=0.8,
                    matched_avoid=["돼지고기"],
                    avoid_evidence=[
                        AvoidEvidence(
                            ingredient="돼지고기",
                            canonical="pork",
                            evidence_type="alias",
                            confidence=0.8,
                            evidence_text="Chorizo",
                        )
                    ],
                )
            ],
            uncertainty_penalty=40,
        )

        result = agent.run(req)

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].menu, "Chorizo Espanol")
        self.assertEqual(result.items[0].menu_original, "Chorizo Espanol")


if __name__ == "__main__":
    unittest.main()
