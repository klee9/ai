import unittest

from app.agents._0_contracts import TranslateInput
from app.agents._0_translate_agent import TranslateAgent


class FakeGemma:
    def __init__(self, text):
        self._text = text

    def generate_text(self, contents, max_output_tokens=1200):
        return self._text


class TranslateAgentTest(unittest.TestCase):
    def test_translate_with_gemma_json(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {"original": "egg", "translated": "달걀", "confidence": 0.95},
                {"original": "milk", "translated": "우유", "confidence": 0.9}
              ]
            }
            """
        )
        agent = TranslateAgent(fake)
        out = agent.run(TranslateInput(texts=["egg", "milk"], source_lang="en", target_lang="ko"))

        self.assertEqual(out.items[0].translated, "달걀")
        self.assertEqual(out.items[0].provider, "gemma_translate")
        self.assertEqual(out.items[1].translated, "우유")

    def test_identity_fallback_when_parse_fails(self):
        fake = FakeGemma("not json")
        agent = TranslateAgent(fake)
        out = agent.run(TranslateInput(texts=["egg"], source_lang="en", target_lang="ko"))

        self.assertEqual(out.items[0].translated, "egg")
        self.assertEqual(out.items[0].provider, "gemma_fallback_parse")

    def test_preserves_duplicate_input_cardinality(self):
        fake = FakeGemma(
            """
            {
              "items": [
                {"original": "egg", "translated": "달걀", "confidence": 0.95},
                {"original": "milk", "translated": "우유", "confidence": 0.9}
              ]
            }
            """
        )
        agent = TranslateAgent(fake)
        out = agent.run(TranslateInput(texts=["egg", "egg", "milk"], source_lang="en", target_lang="ko"))

        self.assertEqual(len(out.items), 3)
        self.assertEqual([it.original for it in out.items], ["egg", "egg", "milk"])
        self.assertEqual([it.translated for it in out.items], ["달걀", "달걀", "우유"])


if __name__ == "__main__":
    unittest.main()
