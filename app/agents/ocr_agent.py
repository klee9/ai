from __future__ import annotations

from typing import Any, List, Optional

from app.agents.contracts import OCRLine, OCROptions, OCROutput

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


class OCRAgent:
    """PaddleOCR로 이미지 텍스트를 뽑는 최소 Agent."""

    def __init__(
        self,
        lang: str = "korean",
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = False,
        use_textline_orientation: bool = False,
        text_det_limit_side_len: int = 960,
    ):
        self.lang = lang
        self.use_doc_orientation_classify = use_doc_orientation_classify
        self.use_doc_unwarping = use_doc_unwarping
        self.use_textline_orientation = use_textline_orientation
        self.text_det_limit_side_len = text_det_limit_side_len
        self._ocr_engine = None

    def run(self, image_bytes: bytes, options: Optional[OCROptions] = None) -> OCROutput:
        opts = options or OCROptions()
        image = self._decode_image(image_bytes)
        if image is None:
            return OCROutput(lines=[], texts=[])

        # PaddleOCR 버전별로 ocr()가 `cls` 인자를 받지 않을 수 있어
        # 최소 호출 형태(이미지만 전달)로 호환성을 유지한다.
        raw = self._get_ocr_engine().ocr(image)
        lines = self._parse_lines(raw)

        # 기본 정리: 공백 제거 + confidence 필터 + 위->아래, 좌->우 정렬
        lines = [
            OCRLine(text=ln.text.strip(), confidence=ln.confidence, bbox=ln.bbox)
            for ln in lines
            if ln.text and ln.text.strip() and ln.confidence >= opts.min_confidence
        ]
        lines.sort(key=lambda x: (self._top_y(x.bbox), self._left_x(x.bbox)))
        return OCROutput(lines=lines, texts=[ln.text for ln in lines])

    def run_from_file(self, image_path: str, options: Optional[OCROptions] = None) -> OCROutput:
        with open(image_path, "rb") as f:
            return self.run(f.read(), options=options)

    def _get_ocr_engine(self):
        if self._ocr_engine is not None:
            return self._ocr_engine

        try:
            from paddleocr import PaddleOCR
        except Exception as exc:
            raise RuntimeError(
                "paddleocr가 설치되지 않았습니다. `pip install paddleocr paddlepaddle` 후 다시 시도하세요."
            ) from exc

        # 속도 우선 설정: 문서 보정/방향 관련 보조 모델은 끄고
        # 텍스트 검출 입력 해상도를 제한해 추론 시간을 줄인다.
        self._ocr_engine = PaddleOCR(
            lang=self.lang,
            use_doc_orientation_classify=self.use_doc_orientation_classify,
            use_doc_unwarping=self.use_doc_unwarping,
            use_textline_orientation=self.use_textline_orientation,
            text_det_limit_side_len=self.text_det_limit_side_len,
        )
        return self._ocr_engine

    @staticmethod
    def _decode_image(image_bytes: bytes):
        if cv2 is None or np is None or not image_bytes:
            return None
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _parse_lines(self, raw: Any) -> List[OCRLine]:
        # 기대 반환:
        # 1) 구버전: [[bbox, (text, conf)], ...] 또는 [[[...], ...]]
        # 2) 신버전: [{"dt_polys": [...], "rec_texts": [...], "rec_scores": [...]}]
        if not isinstance(raw, list) or not raw:
            return []

        # PaddleOCR 신버전(dict 기반) 처리
        if isinstance(raw[0], dict):
            return self._parse_dict_result(raw)

        if self._is_line_item(raw[0]):
            groups = [raw]
        else:
            groups = [g for g in raw if isinstance(g, list)]

        out: List[OCRLine] = []
        for group in groups:
            for item in group:
                line = self._parse_item(item)
                if line is not None:
                    out.append(line)
        return out

    @staticmethod
    def _parse_dict_result(raw: List[Any]) -> List[OCRLine]:
        out: List[OCRLine] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            polys = OCRAgent._first_present(item, ["dt_polys", "rec_polys", "polys", "boxes"], default=[])
            texts = item.get("rec_texts") or []
            scores = item.get("rec_scores") or []

            n = min(len(polys), len(texts), len(scores))
            for i in range(n):
                bbox_raw = polys[i]
                txt_raw = texts[i]
                score_raw = scores[i]

                bbox = OCRAgent._to_bbox_points(bbox_raw)
                text = str(txt_raw or "")
                try:
                    conf = float(score_raw)
                except Exception:
                    conf = 0.0
                conf = max(0.0, min(1.0, conf))
                out.append(OCRLine(text=text, confidence=conf, bbox=bbox))
        return out

    @staticmethod
    def _is_line_item(item: Any) -> bool:
        return isinstance(item, list) and len(item) >= 2 and isinstance(item[0], (list, tuple))

    @staticmethod
    def _parse_item(item: Any) -> Optional[OCRLine]:
        if not isinstance(item, list) or len(item) < 2:
            return None

        bbox_raw, txt_raw = item[0], item[1]
        if not isinstance(bbox_raw, (list, tuple)):
            return None

        bbox = []
        for pt in bbox_raw:
            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                try:
                    bbox.append([float(pt[0]), float(pt[1])])
                except Exception:
                    continue

        text = ""
        conf = 0.0
        if isinstance(txt_raw, (list, tuple)) and len(txt_raw) >= 1:
            text = str(txt_raw[0] or "")
            if len(txt_raw) >= 2:
                try:
                    conf = float(txt_raw[1])
                except Exception:
                    conf = 0.0
        else:
            text = str(txt_raw or "")

        conf = max(0.0, min(1.0, conf))
        return OCRLine(text=text, confidence=conf, bbox=bbox)

    @staticmethod
    def _first_present(data: dict, keys: List[str], default: Any):
        for k in keys:
            if k not in data:
                continue
            v = data.get(k)
            if v is None:
                continue
            return v
        return default

    @staticmethod
    def _to_bbox_points(bbox_raw: Any) -> List[List[float]]:
        """
        PaddleOCR 버전에 따라 bbox가 list/tuple/ndarray 형태로 올 수 있어
        2차원 점 배열이면 모두 [[x,y], ...]로 정규화한다.
        """
        if bbox_raw is None:
            return []

        pts_src = bbox_raw
        if np is not None and hasattr(bbox_raw, "tolist"):
            try:
                pts_src = bbox_raw.tolist()
            except Exception:
                pts_src = bbox_raw

        if not isinstance(pts_src, (list, tuple)):
            return []

        pts: List[List[float]] = []
        for pt in pts_src:
            cur = pt
            if np is not None and hasattr(cur, "tolist"):
                try:
                    cur = cur.tolist()
                except Exception:
                    pass
            if not isinstance(cur, (list, tuple)) or len(cur) < 2:
                continue
            try:
                x = float(cur[0])
                y = float(cur[1])
            except Exception:
                continue
            pts.append([x, y])
        return pts

    @staticmethod
    def _top_y(bbox: List[List[float]]) -> float:
        return min((p[1] for p in bbox), default=0.0)

    @staticmethod
    def _bottom_y(bbox: List[List[float]]) -> float:
        return max((p[1] for p in bbox), default=0.0)

    @staticmethod
    def _left_x(bbox: List[List[float]]) -> float:
        return min((p[0] for p in bbox), default=0.0)
