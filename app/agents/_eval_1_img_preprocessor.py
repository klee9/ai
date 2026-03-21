from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import cv2

class ImagePreprocessAgent:
    """
    메뉴판 이미지 전처리 최소 파이프라인:
    decode -> optional perspective correction
    """

    def __init__(
        self,
        min_short_edge: int = 900,
        enable_perspective: bool = True,
        auto_tune: bool = True,
        ocr_mode: bool = True,
    ):
        # 기존 호출부 호환성을 위해 인자는 유지한다.
        self.min_short_edge = max(256, int(min_short_edge))
        self.enable_perspective = bool(enable_perspective)
        self.auto_tune = bool(auto_tune)
        self.ocr_mode = bool(ocr_mode)

    def run(self, data: bytes, mime_type: str, save_path: Optional[str] = None) -> Tuple[bytes, str]:
        normalized_mime = self._normalize_mime(mime_type)
        if not data:
            return data, normalized_mime

        image = self._decode_image(data)
        if image is None:
            return data, normalized_mime

        transformed = False

        # 1) 문서 사각형 검출 시에만 원근 보정
        if self.enable_perspective:
            quad = self._detect_document_quad(image)
            if quad is not None:
                image = self._warp_from_quad(image, quad)
                transformed = True

        if save_path:
            self._save_local_image(image, save_path)

        if not transformed:
            return data, normalized_mime

        encoded = self._encode_image(image, normalized_mime)
        if encoded is None:
            return data, normalized_mime
        return encoded, normalized_mime

    @staticmethod
    def _normalize_mime(mime_type: str) -> str:
        # 타입 관련
        supported = {"image/jpeg", "image/png", "image/webp"}
        return mime_type if mime_type in supported else "image/png"

    @staticmethod
    def _decode_image(data: bytes) -> Optional["np.ndarray"]:
        buf = np.frombuffer(data, dtype=np.uint8)
        if buf.size == 0:
            return None
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return None
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def _detect_document_quad(self, image: "np.ndarray") -> Optional["np.ndarray"]:
        # 큰 사각 외곽이 잡힐 때만 문서 원근 보정을 수행한다.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(blurred, 50, 150)

        contours, _ = cv2.findContours(edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        img_area = float(image.shape[0] * image.shape[1])
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
            if len(approx) != 4:
                continue
            area_ratio = cv2.contourArea(approx) / max(img_area, 1.0)
            if area_ratio < 0.2 or area_ratio > 0.98:
                continue
            return approx.reshape(4, 2).astype("float32")
        return None

    @staticmethod
    def _warp_from_quad(image: "np.ndarray", quad: "np.ndarray") -> "np.ndarray":
        # 사각형 외곽을 정렬해 촬영 각도 왜곡을 줄인다.
        rect = ImagePreprocessAgent._order_points(quad)
        (tl, tr, br, bl) = rect

        width_a = np.linalg.norm(br - bl)
        width_b = np.linalg.norm(tr - tl)
        height_a = np.linalg.norm(tr - br)
        height_b = np.linalg.norm(tl - bl)

        max_width = int(max(width_a, width_b))
        max_height = int(max(height_a, height_b))
        if max_width < 10 or max_height < 10:
            return image

        dst = np.array(
            [
                [0, 0],
                [max_width - 1, 0],
                [max_width - 1, max_height - 1],
                [0, max_height - 1],
            ],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(rect, dst)
        return cv2.warpPerspective(image, matrix, (max_width, max_height))

    @staticmethod
    def _order_points(pts: "np.ndarray") -> "np.ndarray":
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1).reshape(-1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    @staticmethod
    def _flatten_illumination(image: "np.ndarray") -> "np.ndarray":
        # 저주파 조명 성분(그림자)을 배경으로 추정해 나눗셈 보정한다.
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        short_edge = min(gray.shape[:2])
        k = max(31, (short_edge // 12) | 1)
        background = cv2.GaussianBlur(gray, (k, k), 0)
        flat = cv2.divide(gray, background, scale=255)
        return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _to_grayscale(image: "np.ndarray") -> "np.ndarray":
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _apply_clahe(gray: "np.ndarray") -> "np.ndarray":
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    @staticmethod
    def _save_local_image(image: "np.ndarray", save_path: str) -> None:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), image)

    @staticmethod
    def _encode_image(image: "np.ndarray", mime_type: str) -> Optional[bytes]:
        ext_map = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }
        ext = ext_map[mime_type]
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95] if mime_type == "image/jpeg" else []
        ok, encoded = cv2.imencode(ext, image, params)
        if not ok:
            return None
        return encoded.tobytes()
