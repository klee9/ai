from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    import numpy as np
except Exception:  # pragma: no cover - 환경에 따라 Numpy가 없을 수 있음
    np = None

try:
    import cv2
except Exception:  # pragma: no cover - 환경에 따라 OpenCV가 없을 수 있음
    cv2 = None


class ImagePreprocessAgent:
    """
    OCR/메뉴 추출 전에 이미지 품질을 정규화하는 전처리 Agent.
    처리 순서:
    1) (선택) 너무 작은 이미지일 때만 업스케일
    2) grayscale
    3) CLAHE
    """

    def __init__(self, min_short_edge: int = 900):
        # 짧은 변이 이 값보다 작을 때만 업스케일한다.
        self.min_short_edge = max(256, int(min_short_edge))

    def run(self, data: bytes, mime_type: str, save_path: Optional[str] = None) -> Tuple[bytes, str]:
        """
        입력 이미지 bytes를 전처리해 다시 bytes로 반환한다.
        save_path가 주어지면 전처리 결과 이미지를 로컬 파일로 저장한다.
        전처리가 불가능한 경우(디코딩 실패/OpenCV 없음 등) 원본을 그대로 반환한다.
        """
        normalized_mime = self._normalize_mime(mime_type)
        if not data:
            return data, normalized_mime

        if cv2 is None or np is None:
            # OpenCV/Numpy 미설치 환경에서는 파이프라인 안정성을 위해 원본 통과.
            return data, normalized_mime

        image = self._decode_image(data)
        if image is None:
            # 테스트의 가짜 바이트/손상 이미지에도 예외 없이 동작하도록 폴백.
            return data, normalized_mime

        # 1) 작은 이미지에서만 최소 해상도를 보장하기 위해 업스케일한다.
        resized = self._resize_if_too_small(image)

        # 2) 컬러 노이즈 영향을 줄이기 위해 grayscale로 변환한다.
        gray = self._to_grayscale(resized)

        # 3) 국소 대비를 끌어올리기 위해 CLAHE를 적용한다.
        enhanced = self._apply_clahe(gray)

        # 후속 모델 입력 호환성을 위해 3채널(BGR)로 맞춘다.
        contrasted = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        if save_path:
            self._save_local_image(contrasted, save_path)

        encoded = self._encode_image(contrasted, normalized_mime)
        if encoded is None:
            return data, normalized_mime
        return encoded, normalized_mime

    @staticmethod
    def _normalize_mime(mime_type: str) -> str:
        supported = {"image/jpeg", "image/png", "image/webp"}
        return mime_type if mime_type in supported else "image/png"

    @staticmethod
    def _decode_image(data: bytes) -> Optional[np.ndarray]:
        # bytes -> OpenCV BGR 이미지로 변환한다.
        if np is None:
            return None
        buf = np.frombuffer(data, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _resize_if_too_small(self, image: np.ndarray) -> np.ndarray:
        # 짧은 변이 기준보다 작을 때만 비율을 유지한 채 업스케일한다.
        height, width = image.shape[:2]
        short_edge = min(height, width)
        if short_edge >= self.min_short_edge:
            return image

        scale = self.min_short_edge / float(short_edge)
        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def _to_grayscale(image: np.ndarray) -> np.ndarray:
        # OCR 대상 텍스트 대비를 안정적으로 만들기 위해 회색조로 변환한다.
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _apply_clahe(gray: np.ndarray) -> np.ndarray:
        # CLAHE로 지역 대비를 향상해 흐린 글자의 경계를 강화한다.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _save_local_image(image: np.ndarray, save_path: str) -> None:
        # 개발 중 결과 확인을 위해 전처리 산출 이미지를 파일로 저장한다.
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)

    @staticmethod
    def _encode_image(image: np.ndarray, mime_type: str) -> Optional[bytes]:
        # 전처리 결과를 입력 포맷 기준으로 다시 인코딩한다.
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }
        ext = ext_map[mime_type]

        params = []
        if mime_type == "image/jpeg":
            params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]

        ok, encoded = cv2.imencode(ext, image, params)
        if not ok:
            return None
        return encoded.tobytes()
