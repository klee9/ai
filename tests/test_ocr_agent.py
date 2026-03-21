import unittest
from unittest.mock import patch

from app.agents._0_contracts import OCRLanguageCandidate, OCROptions
from app.agents._eval_2_ocr import OCRAgent


class _FakeEngine:
    def __init__(self, raw_result):
        self.raw_result = raw_result

    def ocr(self, image, cls=True):
        return self.raw_result


class OCRAgentTest(unittest.TestCase):
    def test_sort_candidates_prefers_spanish_when_scores_tie(self):
        candidates = [
            OCRLanguageCandidate(lang="en", score=1.0, line_count=80, avg_confidence=0.93, script_ratio=1.0),
            OCRLanguageCandidate(lang="es", score=1.0, line_count=80, avg_confidence=0.93, script_ratio=1.0),
        ]
        ordered = OCRAgent._sort_candidates(candidates)
        self.assertEqual([item.lang for item in ordered], ["es", "en"])

    def test_default_auto_langs_only_include_korean_english_spanish(self):
        self.assertEqual(OCRAgent.DEFAULT_AUTO_LANGS, ("korean", "en", "es"))

    def test_probe_languages_filter_out_unsupported_langs(self):
        langs = OCRAgent._normalize_probe_languages(["korean", "ch", "en", "es", "japan"])
        self.assertEqual(langs, ["korean", "en", "es"])

    def test_resolve_lang_only_accepts_supported_overrides(self):
        self.assertEqual(OCRAgent._resolve_lang(menu_country_code=None, ocr_lang_override="ko"), "korean")
        self.assertEqual(OCRAgent._resolve_lang(menu_country_code=None, ocr_lang_override="en"), "en")
        self.assertEqual(OCRAgent._resolve_lang(menu_country_code=None, ocr_lang_override="es"), "es")
        self.assertIsNone(OCRAgent._resolve_lang(menu_country_code=None, ocr_lang_override="ch"))
        self.assertIsNone(OCRAgent._resolve_lang(menu_country_code="CN", ocr_lang_override=None))

    def _dummy_image_bytes(self):
        try:
            cv2 = __import__("cv2")
            np = __import__("numpy")
        except Exception:
            self.skipTest("OpenCV/Numpy not available in this environment")

        img = np.full((80, 160, 3), 255, dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", img)
        self.assertTrue(ok)
        return encoded.tobytes()

    def test_keeps_lines_without_multiline_merge(self):
        raw = [
            [[[0, 0], [120, 0], [120, 20], [0, 20]], ("Spicy Pork", 0.95)],
            [[[2, 24], [130, 24], [130, 44], [2, 44]], ("Rice Bowl", 0.93)],
        ]
        agent = OCRAgent()
        data = self._dummy_image_bytes()

        with patch.object(agent, "_get_ocr_engine", return_value=_FakeEngine(raw)):
            out = agent.run(data, options=OCROptions())

        self.assertEqual(out.texts, ["Spicy Pork", "Rice Bowl"])
        self.assertEqual(len(out.lines), 2)

    def test_filters_out_low_confidence_lines(self):
        raw = [
            [[[0, 0], [120, 0], [120, 20], [0, 20]], ("A", 0.91)],
            [[[0, 25], [120, 25], [120, 45], [0, 45]], ("B", 0.20)],
        ]
        agent = OCRAgent()
        data = self._dummy_image_bytes()

        with patch.object(agent, "_get_ocr_engine", return_value=_FakeEngine(raw)):
            out = agent.run(data, options=OCROptions(min_confidence=0.5))

        self.assertEqual(out.texts, ["A"])
        self.assertEqual(len(out.lines), 1)

    def test_parses_new_dict_style_output(self):
        raw = [
            {
                "dt_polys": [
                    [[0, 0], [100, 0], [100, 20], [0, 20]],
                    [[0, 30], [100, 30], [100, 50], [0, 50]],
                ],
                "rec_texts": ["Menu A", "Menu B"],
                "rec_scores": [0.91, 0.88],
            }
        ]
        agent = OCRAgent()
        data = self._dummy_image_bytes()

        with patch.object(agent, "_get_ocr_engine", return_value=_FakeEngine(raw)):
            out = agent.run(data, options=OCROptions(min_confidence=0.5))

        self.assertEqual(out.texts, ["Menu A", "Menu B"])
        self.assertEqual(len(out.lines), 2)
        self.assertEqual(out.lines[0].bbox, [[0.0, 0.0], [100.0, 0.0], [100.0, 20.0], [0.0, 20.0]])

    def test_parses_numpy_polys_in_dict_style_output(self):
        try:
            np = __import__("numpy")
        except Exception:
            self.skipTest("numpy not available in this environment")

        raw = [
            {
                "dt_polys": np.array(
                    [
                        [[1, 2], [11, 2], [11, 22], [1, 22]],
                        [[3, 30], [13, 30], [13, 40], [3, 40]],
                    ],
                    dtype=np.float32,
                ),
                "rec_texts": ["A", "B"],
                "rec_scores": [0.91, 0.88],
            }
        ]
        agent = OCRAgent()
        data = self._dummy_image_bytes()

        with patch.object(agent, "_get_ocr_engine", return_value=_FakeEngine(raw)):
            out = agent.run(data, options=OCROptions(min_confidence=0.5))

        self.assertEqual(out.texts, ["A", "B"])
        self.assertEqual(out.lines[0].bbox, [[1.0, 2.0], [11.0, 2.0], [11.0, 22.0], [1.0, 22.0]])


if __name__ == "__main__":
    unittest.main()
