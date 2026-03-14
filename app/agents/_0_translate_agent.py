from app.agents._0_contracts import TranslateInput, TranslateItem, TranslateOutput
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import clamp_float, extract_first_json_object


class TranslateAgent:
    """
    Gemma 기반 번역 Agent.
    - 외부 번역 API 없이 LLM으로 번역
    - 파싱 실패 시 identity fallback(원문 유지)
    """

    def __init__(self, gemma: GemmaClient):
        self.gemma = gemma

    def run(self, request: TranslateInput) -> TranslateOutput:
        # 변경 배경:
        # - 이전 한계점: normalize_list가 중복까지 제거해 입력/출력 길이 매핑이 깨졌다.
        # - 변경 내용: 중복을 보존하는 전용 normalize를 사용해 API cardinality를 유지한다.
        # 입력 정규화는 하되, 중복은 유지해 API 응답 cardinality를 보장한다.
        texts = self._normalize_texts(request.texts, limit=300)
        if not texts:
            return TranslateOutput(items=[], source_lang=request.source_lang, target_lang=request.target_lang)

        # 변경 배경:
        # - 이전 한계점: 중복을 보존하려면 번역 호출 토큰 비용이 불필요하게 증가했다.
        # - 변경 내용: 프롬프트는 unique로 최소화하고, 응답은 원본 순서/길이로 복원한다.
        # 프롬프트에는 unique 텍스트만 전달해 비용을 줄인다.
        unique_texts = []
        seen = set()
        for text in texts:
            if text in seen:
                continue
            seen.add(text)
            unique_texts.append(text)

        prompt = f"""
You are a high-precision translation engine for food/allergy domains.

INPUT
- texts: {unique_texts}
- source_lang: {request.source_lang}
- target_lang: {request.target_lang}

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

        raw = self.gemma.generate_text([prompt], max_output_tokens=1200)
        data = extract_first_json_object(raw) or {}
        parsed = data.get("items", [])
        # 파싱 실패/빈 결과는 즉시 identity fallback 처리(서비스 중단 방지).
        if not isinstance(parsed, list):
            return self._identity_output(texts, request.source_lang, request.target_lang, "gemma_fallback_parse")
        if not parsed:
            return self._identity_output(texts, request.source_lang, request.target_lang, "gemma_fallback_parse")

        by_original = {}
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

        items = []
        for src in texts:
            if src in by_original:
                items.append(by_original[src])
            else:
                # 변경 배경:
                # - 이전 한계점: 일부 항목 누락 시 클라이언트의 인덱스 매핑이 깨졌다.
                # - 변경 내용: 누락 항목은 identity fallback으로 채워 입력 길이를 유지한다.
                # 일부 항목 누락 시에도 입력 길이를 유지해 백엔드 매핑을 단순화한다.
                items.append(
                    TranslateItem(
                        original=src,
                        translated=src,
                        confidence=0.0,
                        provider="gemma_fallback_missing",
                    )
                )

        return TranslateOutput(items=items, source_lang=request.source_lang, target_lang=request.target_lang)

    @staticmethod
    def _identity_output(texts, source_lang: str, target_lang: str, provider: str) -> TranslateOutput:
        return TranslateOutput(
            items=[
                TranslateItem(
                    original=t,
                    translated=t,
                    confidence=0.0,
                    provider=provider,
                )
                for t in texts
            ],
            source_lang=source_lang,
            target_lang=target_lang,
        )

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
