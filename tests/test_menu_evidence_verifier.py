import unittest

from app.agents._0_contracts import RiskItem, RiskSuspect
from app.utils.menu_evidence_verifier import verify_risk_items


class MenuEvidenceVerifierTest(unittest.TestCase):
    def test_drops_weak_inference_without_menu_signal(self):
        item = RiskItem(
            menu="LA Special Steak",
            confidence=0.4,
            suspects=[
                RiskSuspect(
                    canonical="egg",
                    evidence_type="weak_inference",
                    evidence_text=None,
                    reason="maybe binder",
                    confidence=0.3,
                )
            ],
            matched_avoid=[],
            suspected_ingredients=["beef", "egg"],
        )

        out = verify_risk_items([item], avoid_terms=["egg", "milk"], lang="en")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].matched_avoid, [])
        self.assertEqual(out[0].avoid_evidence, [])
        self.assertIn("beef", [s.casefold() for s in out[0].suspected_ingredients])

    def test_keeps_milk_signal_when_menu_has_direct_token(self):
        item = RiskItem(
            menu="Fresh Milk Pasta",
            confidence=0.5,
            suspects=[
                RiskSuspect(
                    canonical="milk",
                    evidence_type="weak_inference",
                    evidence_text=None,
                    reason="milk-based profile",
                    confidence=0.4,
                )
            ],
            matched_avoid=[],
            suspected_ingredients=["milk"],
        )

        out = verify_risk_items([item], avoid_terms=["milk"], lang="en")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].matched_avoid, ["milk"])
        self.assertEqual(len(out[0].avoid_evidence), 1)
        self.assertIn(out[0].avoid_evidence[0].evidence_type, {"direct", "menu_prior", "weak_inference"})

    def test_family_match_kept_as_weak_inference_without_direct_menu_signal(self):
        item = RiskItem(
            menu="Fettuccine Alfredo Pasta",
            confidence=0.7,
            suspects=[
                RiskSuspect(
                    canonical="dairy",
                    evidence_type="menu_prior",
                    evidence_text=None,
                    reason="cream-based pasta profile",
                    confidence=0.8,
                )
            ],
            matched_avoid=[],
            suspected_ingredients=["dairy"],
        )

        out = verify_risk_items([item], avoid_terms=["milk"], lang="en")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].matched_avoid, ["milk"])
        self.assertEqual(len(out[0].avoid_evidence), 1)
        self.assertIn(out[0].avoid_evidence[0].evidence_type, {"menu_prior", "weak_inference"})
        self.assertEqual(out[0].avoid_evidence[0].canonical, "dairy")

    def test_family_match_survives_conflict_filter_when_other_strong_signal_exists(self):
        item = RiskItem(
            menu="Fettuccine Alfredo Pasta",
            confidence=0.7,
            suspects=[
                RiskSuspect(
                    canonical="wheat",
                    evidence_type="alias",
                    evidence_text="Pasta",
                    reason="pasta base",
                    confidence=0.8,
                ),
                RiskSuspect(
                    canonical="dairy",
                    evidence_type="menu_prior",
                    evidence_text=None,
                    reason="alfredo profile",
                    confidence=0.8,
                ),
            ],
            matched_avoid=[],
            suspected_ingredients=["wheat", "dairy"],
        )

        out = verify_risk_items([item], avoid_terms=["milk"], lang="en")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].matched_avoid, ["milk"])
        self.assertEqual(len(out[0].avoid_evidence), 1)
        self.assertEqual(out[0].avoid_evidence[0].canonical, "dairy")

    def test_family_match_uses_soft_fallback_when_signal_is_weak(self):
        item = RiskItem(
            menu="Crunch Chicken Burger",
            confidence=0.7,
            suspects=[
                RiskSuspect(
                    canonical="dairy",
                    evidence_type="menu_prior",
                    evidence_text=None,
                    reason="possible creamy sauce",
                    confidence=0.9,
                )
            ],
            matched_avoid=[],
            suspected_ingredients=["dairy"],
        )

        out = verify_risk_items([item], avoid_terms=["milk"], lang="en")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].matched_avoid, ["milk"])
        self.assertEqual(len(out[0].avoid_evidence), 1)
        self.assertEqual(out[0].avoid_evidence[0].evidence_type, "weak_inference")
        self.assertLessEqual(out[0].avoid_evidence[0].confidence, 0.25)

    def test_menu_signal_fallback_matches_beef_for_steak(self):
        item = RiskItem(
            menu="토시 스테이크 솥밥",
            confidence=0.6,
            suspects=[],
            matched_avoid=[],
            suspected_ingredients=[],
        )

        out = verify_risk_items([item], avoid_terms=["소고기"], lang="ko")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].matched_avoid, ["소고기"])
        self.assertEqual(len(out[0].avoid_evidence), 1)
        self.assertIn(out[0].avoid_evidence[0].evidence_type, {"alias", "menu_prior", "direct"})

    def test_uses_menu_original_when_display_menu_is_localized(self):
        item = RiskItem(
            menu="스페인 초리소",
            menu_original="Chorizo Espanol",
            confidence=0.6,
            suspects=[],
            matched_avoid=[],
            suspected_ingredients=[],
        )

        out = verify_risk_items([item], avoid_terms=["돼지고기"], lang="ko")

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].menu, "Chorizo Espanol")
        self.assertEqual(out[0].menu_original, "Chorizo Espanol")
        self.assertEqual(out[0].matched_avoid, ["돼지고기"])
        self.assertEqual(len(out[0].avoid_evidence), 1)
        self.assertIn(out[0].avoid_evidence[0].evidence_type, {"alias", "menu_prior", "direct"})


if __name__ == "__main__":
    unittest.main()
