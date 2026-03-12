import re
import unicodedata

from app.agents._0_contracts import AvoidEvidence, RiskAssessInput, RiskAssessOutput, RiskItem
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import clamp_float, clamp_int, extract_first_json_object, normalize_list


def norm(s: str) -> str:
    # 메뉴명 비교 시 대소문자/악센트/공백 차이로 매칭이 깨지지 않게 정규화한다.
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s


LOW_CONF_CUTOFF = 0.45


class RiskAssessAgent:
    def __init__(self, gemma: GemmaClient):
        self.gemma = gemma

    def run(self, request: RiskAssessInput) -> RiskAssessOutput:
        # 입력 정리: 중복 제거 + 비정상 타입 제거
        items = normalize_list(request.items)
        avoid = normalize_list(request.avoid, limit=100)

        if not items:
            return RiskAssessOutput(items=[])

        prompt = f"""
You are a food-risk assessor for menu items.

INPUT
- Menu items (use ONLY these, do not add/remove): {items}
- User avoid-ingredients list: {avoid}

TASK
For each menu item:
1) Predict likely key ingredients (suspected_ingredients).
2) Decide which avoid ingredients might be present and provide evidence:
    - avoid_evidence: list of objects
      - ingredient: must be from avoid list
      - evidence_type: one of ["direct", "common_recipe", "uncertain"]
      - note_ko: short Korean note
3) Output:
    - risk: integer 0~100 (used as conservative prior)
    - confidence: float 0.0~1.0 (higher = more confident about your assessment)
    - suspected_ingredients: up to 3 items
    - matched_avoid: only from avoid list (should match avoid_evidence ingredients)
    - why_ko: ONE short Korean sentence based ONLY on avoid_evidence/suspected_ingredients
IMPORTANT
- If you are unsure, still make a best guess BUT set confidence low.
- It is valid to return no match:
  matched_avoid: [] and avoid_evidence: [] when there is no plausible link.
- Output reasons in Korean.
- Do NOT invent unrelated causes (e.g., mention ingredients that are not in avoid_evidence/suspected_ingredients).

For "menu", copy the menu name EXACTLY from the input list (character-by-character). Do not translate or modify.

OUTPUT: Return ONLY valid JSON (no markdown) with this schema:
{{
  "items": [
    {{
      "menu": "string",
      "risk": 0,
      "confidence": 0.0,
      "suspected_ingredients": ["string"],
      "matched_avoid": ["string"],
      "avoid_evidence": [
        {{
          "ingredient": "string",
          "evidence_type": "direct",
          "note_ko": "string"
        }}
      ],
      "why_ko": "string"
    }}
  ]
}}
""".strip()

        text = self.gemma.generate_text([prompt], max_output_tokens=2000)
        data = extract_first_json_object(text)
        if not data or not isinstance(data.get("items"), list):
            raise ValueError(f"RiskAssess parse failed. RAW: {text[:300]}")

        # 변경 배경:
        # - 이전 한계점: 모델이 메뉴명을 약간 바꾸면 입력 메뉴와 매핑이 끊어질 수 있었다.
        # - 변경 내용: normalize key를 기준으로 원본 메뉴명을 복원하는 매핑 테이블을 사용한다.
        # 모델이 메뉴명을 약간 변형해도 원본 입력과 안전하게 매핑하기 위한 테이블
        allowed = {}
        for menu_name in items:
            key = norm(menu_name)
            if key and key not in allowed:
                allowed[key] = menu_name

        # 변경 배경:
        # - 이전 한계점: avoid 매칭이 exact string 비교라 Egg/egg 같은 표기 차이를 놓쳤다.
        # - 변경 내용: normalize 기반 lookup으로 canonical avoid 토큰으로 정규화한다.
        # avoid 성분 매칭은 대소문자/악센트/공백 차이를 허용해 canonical 값으로 정규화한다.
        avoid_lookup = {}
        for ingredient in avoid:
            key = norm(ingredient)
            if key and key not in avoid_lookup:
                avoid_lookup[key] = ingredient

        by_menu = {}

        for it in data["items"]:
            if not isinstance(it, dict):
                continue

            menu_raw = it.get("menu", "")
            if not isinstance(menu_raw, str):
                continue

            key = norm(menu_raw)
            if key not in allowed:
                # 입력에 없는 메뉴는 hallucination 가능성이 높아 제외
                continue
            menu = allowed[key]
            if menu in by_menu:
                # 동일 메뉴가 중복으로 나오면 첫 결과를 사용
                continue

            risk = clamp_int(it.get("risk", 50), 0, 100, 50)
            conf = clamp_float(it.get("confidence", 0.5), 0.0, 1.0, 0.5)

            suspected = it.get("suspected_ingredients", [])
            if not isinstance(suspected, list):
                suspected = []
            suspected = [s for s in suspected if isinstance(s, str)][:3]

            matched = it.get("matched_avoid", [])
            if not isinstance(matched, list):
                matched = []
            # 변경 배경:
            # - 이전 한계점: avoid 문자열이 조금만 달라도 매칭 실패로 근거가 사라졌다.
            # - 변경 내용: matched_avoid를 canonical avoid 값으로 강제 변환한다.
            # avoid 목록에 실제 존재하는 항목만 남긴다(정규화 비교).
            normalized_matched = []
            for m in matched:
                if not isinstance(m, str):
                    continue
                canonical = avoid_lookup.get(norm(m))
                if not canonical or canonical in normalized_matched:
                    continue
                normalized_matched.append(canonical)
            matched = normalized_matched[:20]

            avoid_evidence_raw = it.get("avoid_evidence", [])
            avoid_evidence = []
            if isinstance(avoid_evidence_raw, list):
                for ev in avoid_evidence_raw:
                    if not isinstance(ev, dict):
                        continue
                    ingredient = ev.get("ingredient", "")
                    if not isinstance(ingredient, str):
                        continue
                    canonical = avoid_lookup.get(norm(ingredient))
                    if not canonical:
                        continue
                    # 변경 배경:
                    # - 이전 한계점: evidence ingredient가 avoid와 표기만 다르면 버려졌다.
                    # - 변경 내용: evidence도 canonical avoid 값으로 통일해 정합성을 유지한다.
                    evidence_type = ev.get("evidence_type", "uncertain")
                    if evidence_type not in {"direct", "common_recipe", "uncertain"}:
                        evidence_type = "uncertain"
                    note_ko = ev.get("note_ko", "")
                    if not isinstance(note_ko, str):
                        note_ko = ""
                    avoid_evidence.append(
                        AvoidEvidence(
                            ingredient=canonical,
                            evidence_type=evidence_type,
                            note_ko=note_ko,
                        )
                    )

            # Backward compatibility:
            # avoid_evidence가 없고 confidence가 충분히 높을 때만 matched_avoid를 uncertain 근거로 승격.
            if not avoid_evidence:
                if matched and conf >= 0.6:
                    avoid_evidence = [
                        AvoidEvidence(ingredient=m, evidence_type="uncertain", note_ko="")
                        for m in matched
                    ]
                else:
                    avoid_evidence = []

            # validator 1) evidence와 matched의 정합성 강제
            evidence_ingredients = []
            for ev in avoid_evidence:
                if ev.ingredient not in evidence_ingredients:
                    evidence_ingredients.append(ev.ingredient)
            matched = [m for m in matched if m in evidence_ingredients]
            if not matched and evidence_ingredients:
                matched = evidence_ingredients[:]

            # validator 2) 저신뢰 + 약한근거(uncertain만 존재)는 no-match 처리
            has_strong_evidence = any(
                ev.evidence_type in {"direct", "common_recipe"} for ev in avoid_evidence
            )
            if conf < LOW_CONF_CUTOFF and not has_strong_evidence:
                matched = []
                avoid_evidence = []
                # no-match 상황에서 과도한 prior risk가 들어오면 완화
                risk = min(risk, 35)

            why = it.get("why_ko", "")
            if not isinstance(why, str):
                why = ""
            if not why.strip() and not avoid_evidence:
                why = "기피 재료 근거 부족"
            if not avoid_evidence:
                why = "기피 재료 근거 부족"

            by_menu[menu] = (
                RiskItem(
                    menu=menu,
                    risk=risk,
                    confidence=conf,
                    suspected_ingredients=suspected,
                    matched_avoid=matched,
                    avoid_evidence=avoid_evidence,
                    why_ko=why,
                )
            )

        # 변경 배경:
        # - 이전 한계점: 모델 출력에서 메뉴가 누락되면 최종 랭킹 후보가 줄어들었다.
        # - 변경 내용: 누락 메뉴를 보수 fallback RiskItem으로 채워 입력 cardinality를 보장한다.
        # LLM이 일부 메뉴를 누락해도 입력 cardinality를 보장한다.
        out = []
        for menu_name in items:
            if menu_name in by_menu:
                out.append(by_menu[menu_name])
                continue
            out.append(
                RiskItem(
                    menu=menu_name,
                    risk=60,
                    confidence=0.0,
                    suspected_ingredients=[],
                    matched_avoid=[],
                    avoid_evidence=[],
                    why_ko="위험도 판단 실패(항목 누락 보수적 처리)",
                )
            )

        return RiskAssessOutput(items=out)
