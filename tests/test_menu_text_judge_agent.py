import unittest

from app.agents._0_contracts import OCRLine
from app.agents._eval_3_extractor import OCRMenuJudgeAgent


class _FakeGemma:
    def __init__(self, responses):
        self.responses = list(responses)

    def image_part_from_bytes(self, data, mime_type):
        return {"mime": mime_type, "size": len(data)}

    def generate_text(self, contents, max_output_tokens=900):
        if not self.responses:
            raise RuntimeError("no fake response")
        return self.responses.pop(0)


class OCRMenuJudgeAgentTest(unittest.TestCase):
    def test_marks_menu_with_text_mapping(self):
        fake = _FakeGemma(['{"items":["Acai Bowl"]}'])
        agent = OCRMenuJudgeAgent(fake)
        out = agent.run(["Acai Bowl", "$198"])

        self.assertEqual([it.text for it in out.items], ["Acai Bowl", "$198"])
        self.assertEqual([it.label for it in out.items], ["menu_item", "other"])
        self.assertEqual(out.menu_texts, ["Acai Bowl"])

    def test_supports_image_mode_with_items_array(self):
        fake = _FakeGemma(['{"items":["Kimchi Fried Rice"]}'])
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Kimchi Fried Rice", confidence=1.0, bbox=[]),
            OCRLine(text="12000", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(lines, image_bytes=b"abc", image_mime="image/png", use_image_context=True)

        self.assertEqual([it.label for it in out.items], ["menu_item", "other"])
        self.assertEqual(out.menu_texts, ["Kimchi Fried Rice"])

    def test_missing_items_fallback_to_other(self):
        fake = _FakeGemma(['{"items":[]}'])
        agent = OCRMenuJudgeAgent(fake)
        out = agent.run(["A", "B"])
        self.assertEqual([it.label for it in out.items], ["other", "other"])
        self.assertEqual(out.menu_texts, [])

    def test_recovers_spanish_menu_titles_when_labeled_as_description(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Mussels/Mejillones 18 GF", "label": "description"},'
                '{"index": 1, "text": "Baby Pimiento Rellenos20", "label": "description"},'
                '{"index": 2, "text": "Sauteed with Marinara Sauce.", "label": "description"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Mussels/Mejillones 18 GF", confidence=1.0, bbox=[]),
            OCRLine(text="Baby Pimiento Rellenos20", confidence=1.0, bbox=[]),
            OCRLine(text="Sauteed with Marinara Sauce.", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertIn("Mussels/Mejillones", out.menu_texts)
        self.assertIn("Baby Pimiento Rellenos", out.menu_texts)
        self.assertNotIn("Sauteed with Marinara Sauce", out.menu_texts)

    def test_spanish_rule_discards_noisy_llm_menu_labels(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Mussels/Mejillones 18 GF", "label": "menu_item"},'
                '{"index": 1, "text": "Sauteed with Marinara Sauce.", "label": "menu_item"},'
                '{"index": 2, "text": "PLEASEINFORMSTAFFOFANYFOODALLERGIESBEFOREORDERING", "label": "menu_item"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Mussels/Mejillones 18 GF", confidence=1.0, bbox=[]),
            OCRLine(text="Sauteed with Marinara Sauce.", confidence=1.0, bbox=[]),
            OCRLine(text="PLEASEINFORMSTAFFOFANYFOODALLERGIESBEFOREORDERING", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertEqual(out.menu_texts, ["Mussels/Mejillones"])

    def test_spanish_rule_handles_tags_and_avoids_wrong_parenthesis_merge(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Patatas Bravas11 GFIVG", "label": "description"},'
                '{"index": 1, "text": "Tabla Iberica24", "label": "description"},'
                '{"index": 2, "text": "(Iberico)y Queso", "label": "description"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Patatas Bravas11 GFIVG", confidence=1.0, bbox=[]),
            OCRLine(text="Tabla Iberica24", confidence=1.0, bbox=[]),
            OCRLine(text="(Iberico)y Queso", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertIn("Patatas Bravas", out.menu_texts)
        self.assertIn("Tabla Iberica", out.menu_texts)
        self.assertNotIn("Tabla Iberica (Iberico)y Queso", out.menu_texts)

    def test_spanish_rule_merges_title_continuation_after_comma_price(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Combinacion de Camarones,24", "label": "description"},'
                '{"index": 1, "text": "6 Deep Fried Shrimps with Orange Ginger", "label": "description"},'
                '{"index": 2, "text": "Churrasco,yChorizo", "label": "description"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Combinacion de Camarones,24", confidence=1.0, bbox=[]),
            OCRLine(text="6 Deep Fried Shrimps with Orange Ginger", confidence=1.0, bbox=[]),
            OCRLine(text="Churrasco,yChorizo", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertIn("Combinacion de Camarones Churrasco,yChorizo", out.menu_texts)

    def test_korean_rule_recovers_title_prefix_from_description_line(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "명란 버터 솥밥 짭조름 한 명란 과 고소한 버터 의", "label": "description"},'
                '{"index": 1, "text": "야채 가 가득 들어 있는", "label": "description"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="명란 버터 솥밥 짭조름 한 명란 과 고소한 버터 의", confidence=1.0, bbox=[]),
            OCRLine(text="야채 가 가득 들어 있는", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="ko",
        )

        self.assertIn("명란 버터 솥밥", out.menu_texts)
        self.assertNotIn("야채 가 가득 들어 있는", out.menu_texts)

    def test_spanish_rule_recovers_main_items_from_inline_price_and_split_price_lines(self):
        fake = _FakeGemma(
            [
                '{"line_labels": ['
                '{"index": 0, "text": "Tapas Calientes", "label": "description"},'
                '{"index": 1, "text": "Mussels / Mejillones | 18 GF Sautéed with Marinara Sauce .", "label": "description"},'
                '{"index": 2, "text": "Calamares Fritos | 18 Fried Calamari .", "label": "description"},'
                '{"index": 3, "text": "Chorizo Español | 18 GF Sautéed Sausage with Onions .", "label": "description"},'
                '{"index": 4, "text": "Croquetas De Jamon | 13 Homemade Ham Croquettes .", "label": "description"},'
                '{"index": 5, "text": "Empanadillas", "label": "description"},'
                '{"index": 6, "text": "13", "label": "description"},'
                '{"index": 7, "text": "Combinación de Camarones , | 24 GF Churrasco , y Chorizo", "label": "description"}'
                "]}",
            ]
        )
        agent = OCRMenuJudgeAgent(fake)
        lines = [
            OCRLine(text="Tapas Calientes", confidence=1.0, bbox=[]),
            OCRLine(text="Mussels / Mejillones | 18 GF Sautéed with Marinara Sauce .", confidence=1.0, bbox=[]),
            OCRLine(text="Calamares Fritos | 18 Fried Calamari .", confidence=1.0, bbox=[]),
            OCRLine(text="Chorizo Español | 18 GF Sautéed Sausage with Onions .", confidence=1.0, bbox=[]),
            OCRLine(text="Croquetas De Jamon | 13 Homemade Ham Croquettes .", confidence=1.0, bbox=[]),
            OCRLine(text="Empanadillas", confidence=1.0, bbox=[]),
            OCRLine(text="13", confidence=1.0, bbox=[]),
            OCRLine(text="Combinación de Camarones , | 24 GF Churrasco , y Chorizo", confidence=1.0, bbox=[]),
        ]
        out = agent.run_lines_with_image(
            lines,
            image_bytes=b"abc",
            image_mime="image/png",
            use_image_context=True,
            ocr_lang="es",
        )

        self.assertIn("Mussels / Mejillones", out.menu_texts)
        self.assertIn("Calamares Fritos", out.menu_texts)
        self.assertIn("Chorizo Español", out.menu_texts)
        self.assertIn("Croquetas De Jamon", out.menu_texts)
        self.assertIn("Empanadillas", out.menu_texts)
        self.assertTrue(any(item.startswith("Combinación de Camarones") and "Chorizo" in item for item in out.menu_texts))
        self.assertNotIn("Tapas Calientes", out.menu_texts)


if __name__ == "__main__":
    unittest.main()
