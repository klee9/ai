import html
import os
import re
from typing import Dict, List, Tuple

import requests

from app.agents._0_contracts import TranslateInput, TranslateItem, TranslateOutput
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import clamp_float, extract_first_json_object


class TranslateAgent:
    """
    번역 Agent.
    - 1순위: Google Cloud Translation API
    - 2순위: Gemma 번역
    - 3순위: identity fallback(원문 유지)
    """

    def __init__(self, gemma: GemmaClient):
        self.gemma = gemma
        self.translate_api_key = (
            os.getenv("GOOGLE_TRANSLATE_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or ""
        ).strip()
        self.translate_endpoint = (
            os.getenv("GOOGLE_TRANSLATE_ENDPOINT")
            or "https://translation.googleapis.com/language/translate/v2"
        ).strip()
        self.translate_timeout_sec = self._safe_float_env("GOOGLE_TRANSLATE_TIMEOUT_SEC", default=10.0, min_value=3.0)

    def run(self, request: TranslateInput) -> TranslateOutput:
        # 변경 배경:
        # - 이전 한계점: normalize_list가 중복까지 제거해 입력/출력 길이 매핑이 깨졌다.
        # - 변경 내용: 중복을 보존하는 전용 normalize를 사용해 API cardinality를 유지한다.
        # 입력 정규화는 하되, 중복은 유지해 API 응답 cardinality를 보장한다.
        texts = self._normalize_texts(request.texts, limit=300)
        if not texts:
            return TranslateOutput(items=[], source_lang=request.source_lang, target_lang=request.target_lang)

        source_lang = self._resolve_lang_code(request.source_lang, allow_auto=True)
        target_lang = self._resolve_lang_code(request.target_lang, allow_auto=False) or "en"
        if source_lang and source_lang != "auto" and source_lang == target_lang:
            return TranslateOutput(
                items=[
                    TranslateItem(
                        original=t,
                        translated=t,
                        confidence=1.0,
                        provider="identity_same_lang",
                    )
                    for t in texts
                ],
                source_lang=request.source_lang,
                target_lang=request.target_lang,
            )

        # 변경 배경:
        # - 이전 한계점: 중복을 보존하려면 번역 호출 토큰 비용이 불필요하게 증가했다.
        # - 변경 내용: 프롬프트는 unique로 최소화하고, 응답은 원본 순서/길이로 복원한다.
        # 프롬프트에는 unique 텍스트만 전달해 비용을 줄인다.
        unique_texts: List[str] = []
        seen = set()
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            unique_texts.append(text)

        by_original: Dict[str, TranslateItem] = {}

        # 1) Google Cloud Translation API (primary)
        if self.translate_api_key:
            try:
                by_original.update(
                    self._translate_with_google(
                        texts=unique_texts,
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                )
            except Exception:
                # Google 실패 시 Gemma fallback으로 진행한다.
                pass

        if by_original:
            self._refine_google_outputs_to_target_lang(by_original, target_lang=target_lang)

        # 2) Gemma fallback for missing items
        missing = [src for src in unique_texts if src not in by_original]
        missing_provider = "identity_fallback_missing"
        if missing:
            gemma_map, missing_provider = self._translate_with_gemma(
                texts=missing,
                source_lang=request.source_lang,
                target_lang=request.target_lang,
            )
            by_original.update(gemma_map)

        items = []
        for src in texts:
            if src in by_original:
                items.append(by_original[src])
            else:
                items.append(
                    TranslateItem(
                        original=src,
                        translated=src,
                        confidence=0.0,
                        provider=missing_provider,
                    )
                )

        return TranslateOutput(items=items, source_lang=request.source_lang, target_lang=request.target_lang)

    def _translate_with_gemma(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> Tuple[Dict[str, TranslateItem], str]:
        if not texts:
            return {}, "gemma_fallback_missing"

        prompt = f"""
You are a high-precision translation engine for food/allergy domains.

INPUT
- texts: {texts}
- source_lang: {source_lang}
- target_lang: {target_lang}

TASK
- Translate each text to target_lang.
- Preserve exact meaning for food/allergy terms.
- Keep the original text unchanged in "original".
- Return one item per distinct original text from texts.

RULES
- Do not add explanations, examples, parentheses, or extra words.
- Return short noun phrases only for ingredient/menu terms.
- Keep proper nouns/transliterated dish names when direct translation is unnatural.
- For allergy terms, prefer standard vocabulary in target_lang.
- If uncertain, provide best translation and lower confidence.

OUTPUT JSON ONLY:
{{
  "items": [
    {{
      "original": "string",
      "translated": "string",
      "confidence": 0.95
    }}
  ]
}}
""".strip()

        try:
            raw = self.gemma.generate_text([prompt], max_output_tokens=1200)
        except Exception:
            return {}, "gemma_fallback_error"

        data = extract_first_json_object(raw) or {}
        parsed = data.get("items", [])
        # 파싱 실패/빈 결과는 즉시 identity fallback 처리(서비스 중단 방지).
        if not isinstance(parsed, list):
            return {}, "gemma_fallback_parse"
        if not parsed:
            return {}, "gemma_fallback_parse"

        by_original: Dict[str, TranslateItem] = {}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            original = item.get("original", "")
            translated = item.get("translated", "")
            confidence = clamp_float(item.get("confidence", 0.8), 0.0, 1.0, 0.8)
            if not isinstance(original, str) or not isinstance(translated, str):
                continue
            key = original.strip()
            if not key:
                continue
            # confidence는 신뢰할 수 없는 값이 들어와도 0~1로 강제한다.
            by_original[key] = TranslateItem(
                original=key,
                translated=translated.strip() or key,
                confidence=confidence,
                provider="gemma_translate",
            )

        return by_original, "gemma_fallback_missing"

    def _translate_with_google(
        self,
        texts: List[str],
        source_lang: str,
        target_lang: str,
    ) -> Dict[str, TranslateItem]:
        if not texts:
            return {}
        if not self.translate_api_key:
            return {}

        payload = {
            "q": texts,
            "target": target_lang,
            "format": "text",
        }
        if source_lang and source_lang != "auto":
            payload["source"] = source_lang

        url = f"{self.translate_endpoint}?key={self.translate_api_key}"
        response = requests.post(url, json=payload, timeout=self.translate_timeout_sec)
        response.raise_for_status()
        data = response.json() if response.content else {}

        if not isinstance(data, dict):
            return {}
        api_error = data.get("error")
        if isinstance(api_error, dict):
            message = str(api_error.get("message", "")).strip()
            raise RuntimeError(f"Google Translate failed: {message or 'unknown error'}")

        translations = ((data.get("data") or {}).get("translations") or [])
        if not isinstance(translations, list):
            return {}

        by_original: Dict[str, TranslateItem] = {}
        for idx, src in enumerate(texts):
            if idx >= len(translations):
                break
            entry = translations[idx]
            if not isinstance(entry, dict):
                continue
            translated = html.unescape(str(entry.get("translatedText", "")).strip())
            if not translated:
                continue

            confidence = 0.98 if translated != src else 0.85
            by_original[src] = TranslateItem(
                original=src,
                translated=translated,
                confidence=confidence,
                provider="google_translate_v2",
            )
        return by_original

    def _refine_google_outputs_to_target_lang(self, by_original: Dict[str, TranslateItem], target_lang: str) -> None:
        # 한국어 대상일 때 라틴 문자가 섞인 결과를 한 번 더 Google 번역으로 정제한다.
        if target_lang != "ko":
            return
        if not by_original:
            return

        src_keys: List[str] = []
        refine_inputs: List[str] = []
        for src, item in by_original.items():
            translated = (item.translated or "").strip()
            if self._needs_target_lang_refine(translated=translated, target_lang=target_lang):
                src_keys.append(src)
                refine_inputs.append(translated)

        if not refine_inputs:
            return

        try:
            refined_map = self._translate_with_google(
                texts=refine_inputs,
                source_lang="auto",
                target_lang=target_lang,
            )
        except Exception:
            return

        for src, refine_input in zip(src_keys, refine_inputs):
            refined_item = refined_map.get(refine_input)
            if refined_item is None:
                continue
            refined_text = (refined_item.translated or "").strip()
            if not refined_text:
                continue

            current = by_original.get(src)
            if current is None:
                continue
            by_original[src] = TranslateItem(
                original=src,
                translated=refined_text,
                confidence=max(current.confidence, refined_item.confidence),
                provider="google_translate_v2_refine",
            )

    @staticmethod
    def _needs_target_lang_refine(translated: str, target_lang: str) -> bool:
        text = (translated or "").strip()
        if not text:
            return False
        if target_lang == "ko":
            return bool(re.search(r"[A-Za-z]", text))
        return False

    @staticmethod
    def _normalize_texts(xs, limit: int = 300):
        if not isinstance(xs, list):
            return []

        out = []
        for x in xs:
            if not isinstance(x, str):
                continue
            s = " ".join(x.split()).strip()
            if not s:
                continue
            out.append(s)
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def _resolve_lang_code(lang: str, allow_auto: bool = False) -> str:
        key = (lang or "").strip().lower()
        if not key:
            return "auto" if allow_auto else ""

        aliases = {
            "auto": "auto",
            "ko": "ko",
            "kr": "ko",
            "korean": "ko",
            "en": "en",
            "english": "en",
            "es": "es",
            "spanish": "es",
        }
        resolved = aliases.get(key, key.split("-", 1)[0])
        if not allow_auto and resolved == "auto":
            return ""
        return resolved

    @staticmethod
    def _safe_float_env(name: str, default: float, min_value: float = 0.0) -> float:
        raw = os.getenv(name, "").strip()
        if not raw:
            return default
        try:
            value = float(raw)
        except Exception:
            return default
        return max(min_value, value)
