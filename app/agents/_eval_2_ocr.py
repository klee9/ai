from __future__ import annotations

from typing import Any, List, Optional

import cv2
import numpy as np

from app.agents._0_contracts import OCRLine, OCROptions, OCROutput


class OCRAgent:
    """PaddleOCR text extractor (CPU baseline)."""

    def __init__(
        self,
        menu_country_code: str = "KR",
        ocr_lang_override: Optional[str] = None,
        text_det_limit_side_len: int = 1216,
        text_recognition_batch_size: int = 4,
        cpu_threads: int = 6,
        det_model_name: str = "PP-OCRv5_mobile_det",
    ):
        self.lang = self._resolve_lang(menu_country_code=menu_country_code, ocr_lang_override=ocr_lang_override)
        self.text_det_limit_side_len = max(256, int(text_det_limit_side_len))
        self.text_recognition_batch_size = max(1, int(text_recognition_batch_size))
        self.cpu_threads = max(1, int(cpu_threads))
        self.det_model_name = (det_model_name or "PP-OCRv5_mobile_det").strip()
        self._ocr_engine = None

    def run(self, image_bytes: bytes, options: Optional[OCROptions] = None) -> OCROutput:
        opts = options or OCROptions()
        image = self._decode_image(image_bytes)
        if image is None:
            return OCROutput(lines=[], texts=[])

        raw = self._get_ocr_engine().ocr(image)
        lines = self._parse_lines(raw)
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

        kwargs_v1 = {
            "lang": self.lang,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "text_det_limit_side_len": self.text_det_limit_side_len,
            "text_recognition_batch_size": self.text_recognition_batch_size,
            "device": "cpu",
            "enable_mkldnn": True,
            "cpu_threads": self.cpu_threads,
        }
        if self.lang in {"en", "ch", "chinese_cht"} and self.det_model_name:
            kwargs_v1["text_detection_model_name"] = self.det_model_name

        kwargs_v2 = {
            "lang": self.lang,
            "det_limit_side_len": self.text_det_limit_side_len,
            "rec_batch_num": self.text_recognition_batch_size,
            "use_gpu": False,
            "enable_mkldnn": True,
            "cpu_threads": self.cpu_threads,
            "use_angle_cls": False,
        }

        try:
            self._ocr_engine = PaddleOCR(**kwargs_v1)
            return self._ocr_engine
        except TypeError:
            try:
                self._ocr_engine = PaddleOCR(**kwargs_v2)
                return self._ocr_engine
            except Exception as exc:
                raise RuntimeError(f"PaddleOCR 엔진 초기화 실패: {exc}") from exc
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR 엔진 초기화 실패: {exc}") from exc

    @staticmethod
    def _resolve_lang(menu_country_code: Optional[str], ocr_lang_override: Optional[str]) -> str:
        lang_aliases = {
            "ko": "korean",
            "korean": "korean",
            "ja": "japan",
            "japan": "japan",
            "zh": "ch",
            "zh-cn": "ch",
            "zh-tw": "chinese_cht",
            "ch": "ch",
            "chinese_cht": "chinese_cht",
            "en": "en",
            "es": "es",
        }
        override = (ocr_lang_override or "").strip().lower()
        if override:
            return lang_aliases.get(override, "en")

        country = (menu_country_code or "KR").strip().upper()
        country = country.replace("_", "-").split("-", 1)[0]
        country_to_lang = {
            "KR": "korean",
            "JP": "japan",
            "CN": "ch",
            "TW": "chinese_cht",
            "HK": "chinese_cht",
            "US": "en",
            "GB": "en",
            "ES": "es",
        }
        return country_to_lang.get(country, "en")

    @staticmethod
    def _decode_image(image_bytes: bytes):
        if not image_bytes:
            return None
        buf = np.frombuffer(image_bytes, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)

    def _parse_lines(self, raw: Any) -> List[OCRLine]:
        if not isinstance(raw, list) or not raw:
            return []

        if isinstance(raw[0], dict):
            return self._parse_dict_result(raw)

        groups = [raw] if self._is_line_item(raw[0]) else [g for g in raw if isinstance(g, list)]
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
                bbox = OCRAgent._to_bbox_points(polys[i])
                text = str(texts[i] or "")
                try:
                    conf = float(scores[i])
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

        bbox = OCRAgent._to_bbox_points(item[0])
        if not bbox:
            return None

        txt_raw = item[1]
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
            v = data.get(k)
            if v is not None:
                return v
        return default

    @staticmethod
    def _to_bbox_points(bbox_raw: Any) -> List[List[float]]:
        if bbox_raw is None:
            return []

        pts_src = bbox_raw.tolist() if hasattr(bbox_raw, "tolist") else bbox_raw
        if not isinstance(pts_src, (list, tuple)):
            return []

        pts: List[List[float]] = []
        for pt in pts_src:
            cur = pt.tolist() if hasattr(pt, "tolist") else pt
            if not isinstance(cur, (list, tuple)) or len(cur) < 2:
                continue
            try:
                pts.append([float(cur[0]), float(cur[1])])
            except Exception:
                continue
        return pts

    @staticmethod
    def _top_y(bbox: List[List[float]]) -> float:
        return min((p[1] for p in bbox), default=0.0)

    @staticmethod
    def _left_x(bbox: List[List[float]]) -> float:
        return min((p[0] for p in bbox), default=0.0)
