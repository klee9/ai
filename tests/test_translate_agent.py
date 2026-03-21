import os
import unittest
from unittest.mock import Mock, patch

import requests

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
        with patch.dict(os.environ, {"GOOGLE_TRANSLATE_API_KEY": ""}, clear=False):
            agent = TranslateAgent(fake)
            out = agent.run(TranslateInput(texts=["egg", "milk"], source_lang="en", target_lang="ko"))

        self.assertEqual(out.items[0].translated, "달걀")
        self.assertEqual(out.items[0].provider, "gemma_translate")
        self.assertEqual(out.items[1].translated, "우유")

    def test_identity_fallback_when_parse_fails(self):
        fake = FakeGemma("not json")
        with patch.dict(os.environ, {"GOOGLE_TRANSLATE_API_KEY": ""}, clear=False):
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
        with patch.dict(os.environ, {"GOOGLE_TRANSLATE_API_KEY": ""}, clear=False):
            agent = TranslateAgent(fake)
            out = agent.run(TranslateInput(texts=["egg", "egg", "milk"], source_lang="en", target_lang="ko"))

        self.assertEqual(len(out.items), 3)
        self.assertEqual([it.original for it in out.items], ["egg", "egg", "milk"])
        self.assertEqual([it.translated for it in out.items], ["달걀", "달걀", "우유"])

    @patch("app.agents._0_translate_agent.requests.post")
    def test_google_translate_primary_path(self, mock_post):
        fake = FakeGemma("not json")

        response = Mock()
        response.raise_for_status.return_value = None
        response.content = b"ok"
        response.json.return_value = {
            "data": {
                "translations": [
                    {"translatedText": "달걀"},
                    {"translatedText": "우유"},
                ]
            }
        }
        mock_post.return_value = response

        with patch.dict(
            os.environ,
            {
                "GOOGLE_TRANSLATE_API_KEY": "dummy-key",
                "GOOGLE_TRANSLATE_TIMEOUT_SEC": "3",
            },
            clear=False,
        ):
            agent = TranslateAgent(fake)
            out = agent.run(TranslateInput(texts=["egg", "milk"], source_lang="en", target_lang="ko"))

        self.assertEqual([it.translated for it in out.items], ["달걀", "우유"])
        self.assertTrue(all(it.provider == "google_translate_v2" for it in out.items))
        self.assertEqual(mock_post.call_count, 1)

    @patch("app.agents._0_translate_agent.requests.post")
    def test_google_error_falls_back_to_gemma(self, mock_post):
        fake = FakeGemma(
            """
            {
              "items": [
                {"original": "egg", "translated": "달걀", "confidence": 0.95}
              ]
            }
            """
        )
        mock_post.side_effect = requests.RequestException("network blocked")

        with patch.dict(
            os.environ,
            {
                "GOOGLE_TRANSLATE_API_KEY": "dummy-key",
                "GOOGLE_TRANSLATE_TIMEOUT_SEC": "3",
            },
            clear=False,
        ):
            agent = TranslateAgent(fake)
            out = agent.run(TranslateInput(texts=["egg"], source_lang="en", target_lang="ko"))

        self.assertEqual(out.items[0].translated, "달걀")
        self.assertEqual(out.items[0].provider, "gemma_translate")

    def test_identity_when_source_and_target_are_same(self):
        fake = FakeGemma("not json")
        with patch.dict(os.environ, {"GOOGLE_TRANSLATE_API_KEY": ""}, clear=False):
            agent = TranslateAgent(fake)
            out = agent.run(TranslateInput(texts=["egg"], source_lang="en", target_lang="en"))

        self.assertEqual(out.items[0].translated, "egg")
        self.assertEqual(out.items[0].provider, "identity_same_lang")

    @patch("app.agents._0_translate_agent.requests.post")
    def test_uses_source_based_override_instead_of_retranslating_google_output(self, mock_post):
        fake = FakeGemma("not json")

        response = Mock()
        response.raise_for_status.return_value = None
        response.content = b"ok"
        response.json.return_value = {
            "data": {
                "translations": [
                    {"translatedText": "duck empanadas"},
                ]
            }
        }
        mock_post.return_value = response

        with patch.dict(
            os.environ,
            {
                "GOOGLE_TRANSLATE_API_KEY": "dummy-key",
            },
            clear=False,
        ):
            agent = TranslateAgent(fake)
            out = agent.run(TranslateInput(texts=["Duck Empanadas"], source_lang="en", target_lang="ko"))

        self.assertEqual(out.items[0].translated, "오리 엠파나다")
        self.assertEqual(out.items[0].provider, "google_translate_v2_override")
        self.assertEqual(mock_post.call_count, 1)

    @patch("app.agents._0_translate_agent.requests.post")
    def test_unresolved_google_mixed_output_falls_back_to_gemma_from_original(self, mock_post):
        fake = FakeGemma(
            """
            {
              "items": [
                {"original": "Unknown Dish", "translated": "알 수 없는 요리", "confidence": 0.92}
              ]
            }
            """
        )
        response = Mock()
        response.raise_for_status.return_value = None
        response.content = b"ok"
        response.json.return_value = {
            "data": {
                "translations": [
                    {"translatedText": "unknown dish"},
                ]
            }
        }
        mock_post.return_value = response

        with patch.dict(
            os.environ,
            {
                "GOOGLE_TRANSLATE_API_KEY": "dummy-key",
            },
            clear=False,
        ):
            agent = TranslateAgent(fake)
            out = agent.run(TranslateInput(texts=["Unknown Dish"], source_lang="en", target_lang="ko"))

        self.assertEqual(out.items[0].translated, "알 수 없는 요리")
        self.assertEqual(out.items[0].provider, "gemma_translate")


if __name__ == "__main__":
    unittest.main()
