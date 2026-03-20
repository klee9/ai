import os
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.agents._0_contracts import FinalResponse, ScoredItem
from app.agents._0_orchestrator import ImageLoadError


class DummyOrchestrator:
    def __init__(self, result=None, exc=None):
        self._result = result
        self._exc = exc

    def run(self, image_url, avoid, user_lang="ko", menu_country_code="AUTO", presigned_url=""):
        if self._exc is not None:
            raise self._exc
        return self._result


class ApiRankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("GOOGLE_API_KEY", "dummy-key")
        import app.api as api_module

        cls.api_module = api_module
        cls.client = TestClient(api_module.app)

    def test_rank_success(self):
        fake_result = FinalResponse(
            items_extracted=["Katsu Don"],
            items=[
                ScoredItem(
                    menu="Katsu Don",
                    score=42,
                    risk=58,
                    confidence=1.0,
                    matched_avoid=["egg"],
                    suspected_ingredients=["egg", "pork"],
                    reason="계란 가능성",
                )
            ],
            best=ScoredItem(
                menu="Katsu Don",
                score=42,
                risk=58,
                confidence=1.0,
                matched_avoid=["egg"],
                suspected_ingredients=["egg", "pork"],
                reason="계란 가능성",
            ),
        )

        with patch.object(self.api_module, "orchestrator", DummyOrchestrator(result=fake_result)):
            res = self.client.post(
                "/rank",
                json={"image_url": "https://example.com/menu.png", "avoid": ["egg"]},
            )

        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(body["items_extracted"], ["Katsu Don"])
        self.assertEqual(body["best"]["menu"], "Katsu Don")
        self.assertEqual(body["items"][0]["score"], 42)

    def test_rank_image_load_error(self):
        with patch.object(
            self.api_module,
            "orchestrator",
            DummyOrchestrator(exc=ImageLoadError("failed to load image from url: bad-url")),
        ):
            res = self.client.post("/rank", json={"image_url": "bad-url", "avoid": []})

        self.assertEqual(res.status_code, 400)
        body = res.json()
        self.assertEqual(body["detail"]["code"], "IMAGE_LOAD_FAILED")

    def test_rank_pipeline_error(self):
        with patch.object(
            self.api_module,
            "orchestrator",
            DummyOrchestrator(exc=RuntimeError("unexpected failure")),
        ):
            res = self.client.post("/rank", json={"image_url": "https://example.com/menu.png", "avoid": []})

        self.assertEqual(res.status_code, 502)
        body = res.json()
        self.assertEqual(body["detail"]["code"], "RANK_PIPELINE_FAILED")


if __name__ == "__main__":
    unittest.main()
