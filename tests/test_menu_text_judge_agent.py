import unittest

from app.agents.contracts import OCRLine
from app.agents.menu_text_judge_agent import OCRMenuJudgeAgent


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


if __name__ == "__main__":
    unittest.main()
