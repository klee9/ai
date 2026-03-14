from app.agents._0_contracts import AvoidIntakeInput, AvoidIntakeOutput
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import extract_first_json_object, normalize_list


class AvoidIntakeAgent:
    """
    자유형 문장에서 기피 재료 후보를 추출하고, 확인 질문 문구를 생성한다.
    """

    def __init__(self, gemma: GemmaClient):
        self.gemma = gemma

    def run(self, request: AvoidIntakeInput) -> AvoidIntakeOutput:
        text = (request.user_text or "").strip()
        # 지원 언어 외 값이 들어오면 기본 한국어로 고정한다.
        lang = request.lang if request.lang in {"ko", "en", "cn"} else "ko"
        if not text:
            q = self._fallback_confirm([], lang)
            return AvoidIntakeOutput(candidates=[], confirm_question=q, confirm_question_ko=q)

        prompt = f"""
You are an allergy-intake extractor.

INPUT
- user_text: {text}
- output_language: {lang}

TASK
1) Extract only food ingredients the user wants to avoid from user_text.
2) Return concise confirmation question in output_language.

RULES
- Extract ingredient nouns only (no full sentence).
- Keep original language tokens as-is.
- Remove duplicates.
- If uncertain, keep candidate list empty.

OUTPUT JSON ONLY:
{{
  "candidates": ["egg", "milk"],
  "confirm_question": "Just to confirm, Are you avoiding egg and milk?"
}}
""".strip()

        raw = self.gemma.generate_text([prompt], max_output_tokens=500)
        # JSON 파싱 실패 시 빈 dict로 처리해 예외 대신 폴백 문구를 사용한다.
        data = extract_first_json_object(raw) or {}
        # 후보 재료는 문자열/중복/공백 정리를 통과한 값만 사용한다.
        candidates = normalize_list(data.get("candidates", []), limit=50)
        confirm = data.get("confirm_question", "")
        if not isinstance(confirm, str) or not confirm.strip():
            confirm = self._fallback_confirm(candidates, lang)

        confirm = confirm.strip()
        return AvoidIntakeOutput(candidates=candidates, confirm_question=confirm, confirm_question_ko=confirm)

    @staticmethod
    def _fallback_confirm(candidates, lang: str) -> str:
        # LLM 응답이 비정상이거나 비어 있을 때, 최소 UX를 보장하는 규칙 기반 질문 생성.
        if not candidates:
            return {
                "ko": "기피 재료를 다시 알려줘.",
                "en": "Please tell me your avoid ingredients again.",
                "cn": "请再告诉我你要忌口的食材。",
            }[lang]

        joined = ", ".join(candidates)
        return {
            "ko": f"너 {joined} 피하는 거 맞지?",
            "en": f"Just to confirm, are you avoiding {joined}?",
            "cn": f"确认一下，你是要避免 {joined} 吗？",
        }[lang]
