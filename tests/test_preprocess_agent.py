import os
import tempfile
import unittest

from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent


class ImagePreprocessAgentTest(unittest.TestCase):
    def test_returns_original_when_decode_fails(self):
        agent = ImagePreprocessAgent()
        raw = b"not-an-image"

        out_data, out_mime = agent.run(raw, "image/png")

        self.assertEqual(out_data, raw)
        self.assertEqual(out_mime, "image/png")

    def test_supports_minimal_valid_image_input(self):
        try:
            cv2 = __import__("cv2")
            np = __import__("numpy")
        except Exception:
            self.skipTest("OpenCV/Numpy not available in this environment")

        # 작은 합성 이미지를 만들어 전처리 파이프라인이 예외 없이 동작하는지 검증한다.
        img = np.full((120, 180, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (160, 100), (230, 230, 230), thickness=-1)
        cv2.putText(img, "MENU", (35, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        ok, encoded = cv2.imencode(".png", img)
        self.assertTrue(ok)

        agent = ImagePreprocessAgent()
        out_data, out_mime = agent.run(encoded.tobytes(), "image/png")

        self.assertTrue(len(out_data) > 0)
        self.assertEqual(out_mime, "image/png")

    def test_can_save_preprocessed_image_locally(self):
        try:
            cv2 = __import__("cv2")
            np = __import__("numpy")
        except Exception:
            self.skipTest("OpenCV/Numpy not available in this environment")

        img = np.full((120, 180, 3), 255, dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (160, 100), (230, 230, 230), thickness=-1)
        ok, encoded = cv2.imencode(".png", img)
        self.assertTrue(ok)

        agent = ImagePreprocessAgent()
        with tempfile.TemporaryDirectory() as tmp_dir:
            save_path = os.path.join(tmp_dir, "preprocessed.png")
            _, _ = agent.run(encoded.tobytes(), "image/png", save_path=save_path)
            self.assertTrue(os.path.exists(save_path))
            self.assertGreater(os.path.getsize(save_path), 0)


if __name__ == "__main__":
    unittest.main()
