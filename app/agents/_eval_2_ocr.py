from __future__ import annotations

import base64
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from app.agents._0_contracts import OCRLanguageCandidate, OCRLine, OCROptions, OCROutput


class OCRAgent:
    """Google Vision OCR extractor."""

    SUPPORTED_OCR_LANGS: Tuple[str, ...] = ("korean", "en", "es")
    DEFAULT_AUTO_LANGS: Tuple[str, ...] = SUPPORTED_OCR_LANGS
    CJK_LANGS = {"korean"}
    SPANISH_HINT_CHARS = set("áéíóúüñÁÉÍÓÚÜÑ¿¡")

    def __init__(
        self,
        menu_country_code: Optional[str] = "AUTO",
        ocr_lang_override: Optional[str] = None,
        ocr_backend: Optional[str] = None,  # 하위 호환용(무시)
        text_det_limit_side_len: int = 1216,  # 하위 호환용(무시)
        text_recognition_batch_size: int = 4,  # 하위 호환용(무시)
        cpu_threads: int = 6,  # 하위 호환용(무시)
        det_model_name: str = "",  # 하위 호환용(무시)
        probe_languages: Optional[Sequence[str]] = None,  # 하위 호환용
        probe_text_det_limit_side_len: int = 512,  # 하위 호환용(무시)
        probe_text_recognition_batch_size: int = 8,  # 하위 호환용(무시)
        probe_cpu_threads: int = 4,  # 하위 호환용(무시)
        probe_image_max_side: int = 960,  # 하위 호환용(무시)
        probe_center_crop_ratio: float = 0.72,  # 하위 호환용(무시)
        probe_large_image_threshold: int = 1800,  # 하위 호환용(무시)
        probe_refine_top_k: int = 2,  # 하위 호환용(무시)
        probe_early_exit_score_gap: float = 0.12,  # 하위 호환용(무시)
        probe_early_exit_min_score: float = 0.72,  # 하위 호환용(무시)
        probe_early_exit_min_script_ratio: float = 0.6,  # 하위 호환용(무시)
    ):
        _ = (
            ocr_backend,
            text_det_limit_side_len,
            text_recognition_batch_size,
            cpu_threads,
            det_model_name,
            probe_text_det_limit_side_len,
            probe_text_recognition_batch_size,
            probe_cpu_threads,
            probe_image_max_side,
            probe_center_crop_ratio,
            probe_large_image_threshold,
            probe_refine_top_k,
            probe_early_exit_score_gap,
            probe_early_exit_min_score,
            probe_early_exit_min_script_ratio,
        )
        resolved_lang = self._resolve_lang(menu_country_code=menu_country_code, ocr_lang_override=ocr_lang_override)
        self.requested_lang = resolved_lang
        self.lang = resolved_lang or "auto"
        self.lang_source = "manual" if resolved_lang else "auto"
        self.probe_languages = self._normalize_probe_languages(probe_languages)
        self.vision_api_key = (
            os.getenv("GOOGLE_VISION_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        ).strip()
        self.request_type = (os.getenv("OCR_VISION_REQUEST_TYPE") or "DOCUMENT_TEXT_DETECTION").strip().upper()
        self.request_timeout_sec = max(5.0, float(os.getenv("OCR_VISION_TIMEOUT_SEC", "25") or "25"))
        self.last_detection_candidates: List[OCRLanguageCandidate] = []
        self.last_resolved_lang = self.lang
        self.last_lang_source = self.lang_source

    def run(self, image_bytes: bytes, options: Optional[OCROptions] = None) -> OCROutput:
        opts = options or OCROptions()
        if not image_bytes:
            return OCROutput(lines=[], texts=[], resolved_lang="", lang_detection_source="")
        if not self.vision_api_key:
            raise RuntimeError("Google Vision API key is required. Set GOOGLE_VISION_API_KEY (or GOOGLE_API_KEY).")

        data = self._predict(image_bytes)
        lines, locale = self._parse_vision_lines(data)
        lines = self._filter_and_sort_lines(
            lines,
            min_confidence=opts.min_confidence,
            include_bbox=bool(opts.include_bbox),
        )
        resolved_lang, lang_source = self._resolve_run_lang(locale)

        self.lang = resolved_lang
        self.lang_source = lang_source
        self.last_resolved_lang = resolved_lang
        self.last_lang_source = lang_source
        self.last_detection_candidates = []

        return OCROutput(
            lines=lines,
            texts=[ln.text for ln in lines],
            resolved_lang=resolved_lang,
            lang_detection_source=lang_source,
            lang_detection_candidates=[],
        )

    def run_from_file(self, image_path: str, options: Optional[OCROptions] = None) -> OCROutput:
        with open(image_path, "rb") as f:
            return self.run(f.read(), options=options)

    def _resolve_run_lang(self, locale: str) -> Tuple[str, str]:
        if self.requested_lang:
            return self.requested_lang, "manual"

        prefix = (locale or "").strip().lower().split("-", 1)[0]
        mapped = {
            "ko": "korean",
            "en": "en",
            "es": "es",
        }.get(prefix, "")
        if mapped:
            return mapped, "vision_locale"
        return "en", "fallback"

    def _predict(self, image_bytes: bytes) -> Dict[str, Any]:
        content = base64.b64encode(image_bytes).decode("utf-8")
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.vision_api_key}"
        payload = {
            "requests": [
                {
                    "image": {"content": content},
                    "features": [{"type": self.request_type}],
                }
            ]
        }
        response = requests.post(url, json=payload, timeout=self.request_timeout_sec)
        response.raise_for_status()
        data = response.json() if response.content else {}
        if not isinstance(data, dict):
            return {}
        return data

    def _parse_vision_lines(self, data: Dict[str, Any]) -> Tuple[List[OCRLine], str]:
        responses = data.get("responses", []) if isinstance(data, dict) else []
        if not responses or not isinstance(responses[0], dict):
            return [], ""
        response0 = responses[0]
        if isinstance(response0.get("error"), dict):
            message = str(response0["error"].get("message", "")).strip()
            raise RuntimeError(f"Google Vision OCR failed: {message or 'unknown error'}")

        full = response0.get("fullTextAnnotation", {}) if isinstance(response0, dict) else {}
        locale = str(full.get("locale", "")).strip() if isinstance(full, dict) else ""

        lines: List[OCRLine] = []
        pages = full.get("pages", []) if isinstance(full, dict) else []
        for page in pages or []:
            for block in page.get("blocks", []) or []:
                for paragraph in block.get("paragraphs", []) or []:
                    text = self._vision_paragraph_text(paragraph)
                    if not text:
                        continue
                    conf = self._vision_confidence(paragraph, default=0.9)
                    bbox = self._vision_vertices_to_bbox(
                        (paragraph.get("boundingBox", {}) or {}).get("vertices", [])
                    )
                    lines.append(OCRLine(text=text, confidence=conf, bbox=bbox))
        if lines:
            return lines, locale

        text_ann = response0.get("textAnnotations", []) if isinstance(response0, dict) else []
        for ann in (text_ann[1:] if len(text_ann) > 1 else []):
            if not isinstance(ann, dict):
                continue
            text = str(ann.get("description", "")).strip()
            if not text:
                continue
            bbox = self._vision_vertices_to_bbox(
                (ann.get("boundingPoly", {}) or {}).get("vertices", [])
            )
            lines.append(OCRLine(text=text, confidence=0.9, bbox=bbox))
        return lines, locale

    @staticmethod
    def _vision_paragraph_text(paragraph: Dict[str, Any]) -> str:
        words = paragraph.get("words", []) if isinstance(paragraph, dict) else []
        out_words: List[str] = []
        for word in words or []:
            symbols = word.get("symbols", []) if isinstance(word, dict) else []
            token = "".join(str(symbol.get("text", "")) for symbol in symbols if isinstance(symbol, dict))
            token = token.strip()
            if token:
                out_words.append(token)
        return " ".join(out_words).strip()

    @staticmethod
    def _vision_confidence(item: Dict[str, Any], default: float = 0.9) -> float:
        try:
            value = float(item.get("confidence", default))
        except Exception:
            value = float(default)
        return max(0.0, min(1.0, value))

    @staticmethod
    def _vision_vertices_to_bbox(vertices: Any) -> List[List[float]]:
        out: List[List[float]] = []
        if not isinstance(vertices, list):
            return out
        for vertex in vertices:
            if not isinstance(vertex, dict):
                continue
            try:
                x = float(vertex.get("x", 0) or 0)
                y = float(vertex.get("y", 0) or 0)
            except Exception:
                continue
            out.append([x, y])
        return out

    @staticmethod
    def _filter_and_sort_lines(
        lines: List[OCRLine],
        min_confidence: float,
        include_bbox: bool = True,
    ) -> List[OCRLine]:
        filtered = [
            OCRLine(text=ln.text.strip(), confidence=ln.confidence, bbox=ln.bbox)
            for ln in lines
            if ln.text and ln.text.strip() and ln.confidence >= min_confidence
        ]
        filtered.sort(key=lambda x: (OCRAgent._top_y(x.bbox), OCRAgent._left_x(x.bbox)))
        if not include_bbox:
            return [OCRLine(text=ln.text, confidence=ln.confidence, bbox=[]) for ln in filtered]
        return filtered

    @classmethod
    def _expected_script_ratio(cls, texts: List[str], lang: str) -> float:
        profile = cls._character_profile(texts)
        hangul_ratio = profile["hangul_ratio"]
        latin_ratio = profile["latin_ratio"]
        spanish_ratio = profile["spanish_ratio"]

        if lang == "korean":
            return min(1.0, hangul_ratio)
        if lang == "es":
            return min(1.0, latin_ratio + (spanish_ratio * 0.35))
        if lang == "en":
            return max(0.0, min(1.0, latin_ratio - (spanish_ratio * 0.10)))
        return 0.0

    @classmethod
    def _character_profile(cls, texts: List[str]) -> Dict[str, float]:
        joined = "".join(texts)
        signal_chars = 0
        hangul = 0
        latin = 0
        spanish = 0
        for ch in joined:
            code = ord(ch)
            if ch in cls.SPANISH_HINT_CHARS:
                spanish += 1
            if 0xAC00 <= code <= 0xD7A3:
                hangul += 1
                signal_chars += 1
            elif cls._is_latin_char(ch):
                latin += 1
                signal_chars += 1

        denom = float(signal_chars or 1)
        return {
            "hangul_ratio": hangul / denom,
            "latin_ratio": latin / denom,
            "spanish_ratio": min(1.0, spanish / denom),
        }

    @staticmethod
    def _is_latin_char(ch: str) -> bool:
        code = ord(ch)
        return (
            0x0041 <= code <= 0x005A
            or 0x0061 <= code <= 0x007A
            or 0x00C0 <= code <= 0x00FF
            or 0x0100 <= code <= 0x017F
        )

    @classmethod
    def warmup_shared_engines(
        cls,
        langs: Optional[Sequence[str]] = None,
        preload_probe: bool = True,
        preload_full: bool = False,
    ) -> None:
        _ = (langs, preload_probe, preload_full)
        # Vision-only 경로에서는 별도 로컬 엔진 로딩이 없다.
        return None

    @classmethod
    def _normalize_probe_languages(cls, probe_languages: Optional[Sequence[str]]) -> List[str]:
        raw_values = list(probe_languages) if probe_languages else list(cls.DEFAULT_AUTO_LANGS)
        out: List[str] = []
        seen = set()
        for value in raw_values:
            if not isinstance(value, str):
                continue
            lang = value.strip().lower()
            if not lang or lang in seen or lang not in cls.SUPPORTED_OCR_LANGS:
                continue
            seen.add(lang)
            out.append(lang)
        return out or list(cls.DEFAULT_AUTO_LANGS)

    @staticmethod
    def _resolve_lang(menu_country_code: Optional[str], ocr_lang_override: Optional[str]) -> Optional[str]:
        lang_aliases = {
            "auto": None,
            "ko": "korean",
            "korean": "korean",
            "en": "en",
            "es": "es",
        }
        override = (ocr_lang_override or "").strip().lower()
        if override:
            return lang_aliases.get(override)

        country = (menu_country_code or "").strip().upper()
        if not country or country in {"AUTO", "UNKNOWN", "NONE"}:
            return None

        country = country.replace("_", "-").split("-", 1)[0]
        country_to_lang = {
            "KR": "korean",
            "US": "en",
            "GB": "en",
            "AU": "en",
            "CA": "en",
            "ES": "es",
            "MX": "es",
            "AR": "es",
            "CL": "es",
            "CO": "es",
            "PE": "es",
        }
        return country_to_lang.get(country)

    @staticmethod
    def _top_y(bbox: List[List[float]]) -> float:
        return min((p[1] for p in bbox), default=0.0)

    @staticmethod
    def _left_x(bbox: List[List[float]]) -> float:
        return min((p[0] for p in bbox), default=0.0)
