import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agents._0_contracts import AvoidIntakeOutput, TranslateItem, TranslateOutput


class DummyOrchestratorForTranslate:
    def __init__(self, translate_out=None, intake_out=None, exc=None):
        self._translate_out = translate_out
        self._intake_out = intake_out
        self._exc = exc

    def translate_only(self, texts, source_lang, target_lang):
        if self._exc is not None:
            raise self._exc
        return self._translate_out

    def intake_avoid(self, user_text, lang="ko"):
        if self._exc is not None:
            raise self._exc
        return self._intake_out


class ApiTranslateTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("GOOGLE_API_KEY", "dummy-key")
        import app.api as api_module

        cls.api_module = api_module
        cls.client = TestClient(api_module.app)

    def test_translate_success(self):
        fake_translate = TranslateOutput(
            items=[
                TranslateItem(
                    original="egg",
                    translated="달걀",
                    confidence=1.0,
                    provider="google_translate_v2",
                )
            ],
            source_lang="en",
            target_lang="ko",
        )
        dummy = DummyOrchestratorForTranslate(translate_out=fake_translate)
        with patch.object(self.api_module, "orchestrator", dummy):
            res = self.client.post(
                "/translate",
                json={"texts": ["egg"], "source_lang": "en", "target_lang": "ko"},
            )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["items"][0]["translated"], "달걀")

    def test_avoid_intake_success(self):
        fake_out = AvoidIntakeOutput(
            candidates=["egg", "milk"],
            confirm_question="너 egg, milk 피하는 거 맞지?",
            confirm_question_ko="너 egg, milk 피하는 거 맞지?",
        )
        dummy = DummyOrchestratorForTranslate(intake_out=fake_out)
        with patch.object(self.api_module, "orchestrator", dummy):
            res = self.client.post(
                "/avoid/intake",
                json={"user_text": "나는 egg랑 milk 못먹어", "lang": "ko"},
            )
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["candidates"], ["egg", "milk"])
        self.assertIn("맞지", body["confirm_question"])


if __name__ == "__main__":
    unittest.main()
