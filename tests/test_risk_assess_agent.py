import unittest

from app.agents._0_contracts import RiskAssessInput
from app.agents._eval_4_1_risk_assessor import RiskAssessAgent


class FakeGemma:
    def __init__(self, text: str):
        self._text = text
        self.last_prompt = ""

    def generate_text(self, contents, max_output_tokens=2000):
        if isinstance(contents, list) and contents:
            self.last_prompt = str(contents[0])
        return self._text


class RiskAssessAgentTest(unittest.TestCase):
    def test_prompt_uses_global_ingredient_canonicals(self):
        fake = FakeGemma('{"items":[{"menu_name":"LA Special Steak","confidence":0.8,"suspects":[]}]}')
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["LA Special Steak"], avoid=["egg"])

        agent.run(req)

        self.assertIn("Allowed ingredient canonicals", fake.last_prompt)
        self.assertIn('"beef"', fake.last_prompt)
        self.assertIn('"wheat"', fake.last_prompt)

    def test_infers_representative_ingredients_without_forced_avoid_match(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu_name": "LA Special Steak",
                  "confidence": 0.7,
                  "suspects": [
                    {"canonical": "beef", "evidence_type": "menu_prior", "evidence_text": null, "reason": "steak profile", "confidence": 0.8}
                  ]
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["LA Special Steak"], avoid=["egg", "milk"])

        result = agent.run(req)

        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        self.assertEqual(item.menu, "LA Special Steak")
        self.assertEqual(item.suspected_ingredients, ["beef"])
        self.assertEqual(item.matched_avoid, [])
        self.assertEqual(item.avoid_evidence, [])

    def test_matches_child_canonical_to_parent_avoid(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu_name": "Penne Arrabbiata Pasta",
                  "confidence": 0.8,
                  "suspects": [
                    {"canonical": "wheat", "evidence_type": "alias", "evidence_text": "Pasta", "reason": "pasta base", "confidence": 0.8}
                  ]
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Penne Arrabbiata Pasta"], avoid=["gluten"])

        result = agent.run(req)

        self.assertEqual(len(result.items), 1)
        item = result.items[0]
        self.assertEqual(item.matched_avoid, ["gluten"])
        self.assertEqual(len(item.avoid_evidence), 1)
        self.assertEqual(item.avoid_evidence[0].canonical, "wheat")

    def test_matches_parent_canonical_to_child_avoid_conservatively(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu_name": "Fettuccine Alfredo Pasta",
                  "confidence": 0.9,
                  "suspects": [
                    {"canonical": "dairy", "evidence_type": "menu_prior", "evidence_text": null, "reason": "alfredo profile", "confidence": 0.9}
                  ]
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Fettuccine Alfredo Pasta"], avoid=["milk"])

        result = agent.run(req)

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].matched_avoid, ["milk"])
        self.assertEqual(result.items[0].avoid_evidence[0].canonical, "dairy")

    def test_matches_same_family_sibling_canonical(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu_name": "Cheese Pasta",
                  "confidence": 0.8,
                  "suspects": [
                    {"canonical": "cheese", "evidence_type": "menu_prior", "evidence_text": null, "reason": "cheese profile", "confidence": 0.8}
                  ]
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Cheese Pasta"], avoid=["milk"])

        result = agent.run(req)

        self.assertEqual(len(result.items), 1)
        self.assertEqual(result.items[0].matched_avoid, ["milk"])
        self.assertEqual(result.items[0].avoid_evidence[0].canonical, "cheese")

    def test_caps_suspects_to_five_items(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu_name": "Sample Dish",
                  "confidence": 0.8,
                  "suspects": [
                    {"canonical": "egg", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "milk", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "cheese", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "butter", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "peanut", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "tree nut", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "soy", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "wheat", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "shrimp", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "crab", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "fish", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4},
                    {"canonical": "beef", "evidence_type": "weak_inference", "evidence_text": null, "reason": "", "confidence": 0.4}
                  ]
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Sample Dish"], avoid=["egg"])

        result = agent.run(req)

        self.assertEqual(len(result.items), 1)
        self.assertEqual(len(result.items[0].suspected_ingredients), 5)

    def test_fills_missing_menu_items_with_fallback_item(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {
                  "menu_name": "Menu A",
                  "confidence": 0.8,
                  "suspects": []
                }
              ]
            }
            """
        )
        agent = RiskAssessAgent(fake)
        req = RiskAssessInput(items=["Menu A", "Menu B"], avoid=["egg"])

        result = agent.run(req)

        self.assertEqual([it.menu for it in result.items], ["Menu A", "Menu B"])
        self.assertEqual(result.items[1].risk, 0)
        self.assertEqual(result.items[1].confidence, 0.0)
        self.assertEqual(result.items[1].matched_avoid, [])


if __name__ == "__main__":
    unittest.main()
