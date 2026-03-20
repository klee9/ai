import unittest

from app.utils.avoid_ingredient_synonyms import (
    build_avoid_synonym_lookup,
    find_matching_avoid_canonical,
    get_display_name,
    get_canonical_ingredient,
    get_menu_evidence_catalog,
    normalize_ingredient_token,
)


class AvoidIngredientSynonymsTest(unittest.TestCase):
    def test_lookup_excludes_cn_terms_and_keeps_supported_lang_terms(self):
        lookup = build_avoid_synonym_lookup()

        self.assertIn("egg", lookup)
        self.assertIn(normalize_ingredient_token("계란"), lookup)
        self.assertNotIn(normalize_ingredient_token("鸡蛋"), lookup)

    def test_display_name_for_spanish_prefers_es_labels(self):
        self.assertEqual(get_display_name("egg", lang="es"), "huevo")
        self.assertEqual(get_display_name("milk", lang="es"), "leche")

    def test_menu_evidence_catalog_excludes_cn_aliases(self):
        catalog = get_menu_evidence_catalog()
        milk_direct_terms = [term.casefold() for term in catalog["milk"]["direct"]]
        self.assertIn("milk", milk_direct_terms)
        self.assertNotIn("牛奶".casefold(), milk_direct_terms)

    def test_find_matching_canonical_supports_same_family_match(self):
        self.assertEqual(find_matching_avoid_canonical("cheese", {"milk"}), "milk")
        self.assertEqual(find_matching_avoid_canonical("milk", {"cheese"}), "cheese")

    def test_spanish_aliases_map_to_pork(self):
        lookup = build_avoid_synonym_lookup()
        self.assertEqual(lookup[normalize_ingredient_token("chorizo")], "pork")
        self.assertEqual(lookup[normalize_ingredient_token("jamón serrano")], "pork")
        self.assertEqual(get_canonical_ingredient("tabla de serrano", mode="menu_all"), "pork")

    def test_spanish_menu_evidence_catalog_includes_key_pork_terms(self):
        catalog = get_menu_evidence_catalog()
        strong = {normalize_ingredient_token(term) for term in catalog["pork"]["strong"]}
        prior = {normalize_ingredient_token(term) for term in catalog["pork"]["prior"]}
        self.assertIn(normalize_ingredient_token("chorizo"), strong)
        self.assertIn(normalize_ingredient_token("jamon serrano"), strong)
        self.assertIn(normalize_ingredient_token("caldo gallego"), prior)


if __name__ == "__main__":
    unittest.main()
