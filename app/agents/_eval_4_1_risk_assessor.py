import json
import re
import unicodedata

from app.agents._0_contracts import (
    AvoidEvidence,
    RiskAssessInput,
    RiskAssessOutput,
    RiskItem,
    RiskSuspect,
)
from app.clients.gemma_client import GemmaClient
from app.utils.avoid_ingredient_synonyms import canonicalize_avoid_ingredients, get_canonical_ingredient
from app.utils.parsing import clamp_float, extract_first_json_object, normalize_list


def norm(s: str) -> str:
    # 메뉴명 비교 시 대소문자/악센트/공백 차이로 매칭이 깨지지 않게 정규화한다.
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = "".join(ch for ch in unicodedata.normalize("NFKD", s) if not unicodedata.combining(ch))
    return s


DEFAULT_BATCH_SIZE = 15
WEAK_INFERENCE_MAX_CONF = 0.45
SUSPECT_RANK = {
    "direct": 4,
    "alias": 3,
    "menu_prior": 2,
    "weak_inference": 1,
    "none": 0,
}


class RiskAssessAgent:
    def __init__(self, gemma: GemmaClient, batch_size: int = DEFAULT_BATCH_SIZE):
        self.gemma = gemma
        self.batch_size = max(1, int(batch_size))

    def run(self, request: RiskAssessInput) -> RiskAssessOutput:
        # 입력 정리: 중복 제거 + 비정상 타입 제거
        items = normalize_list(request.items)
        avoid = normalize_list(request.avoid, limit=100)

        if not items:
            return RiskAssessOutput(items=[])
        if not avoid:
            # 기피 재료가 없으면 위험 평가 자체가 성립하지 않으므로 LLM 호출을 생략한다.
            return RiskAssessOutput(
                items=[
                    RiskItem(
                        menu=menu_name,
                        risk=0,
                        confidence=1.0,
                        suspected_ingredients=[],
                        suspects=[],
                        matched_avoid=[],
                        avoid_evidence=[],
                    )
                    for menu_name in items
                ]
            )
        if len(items) <= self.batch_size:
            return self._run_batch(items, avoid)

        merged_items = []
        for start in range(0, len(items), self.batch_size):
            batch_items = items[start:start + self.batch_size]
            batch_output = self._run_batch(batch_items, avoid)
            merged_items.extend(batch_output.items)
        return RiskAssessOutput(items=merged_items)

    def _generate_assessment(self, prompt: str):
        text = self.gemma.generate_text([prompt], max_output_tokens=2000)
        data = extract_first_json_object(text)
        if not data or not isinstance(data.get("items"), list):
            raise ValueError(f"RiskAssess parse failed. RAW: {text[:300]}")
        return data

    @staticmethod
    def _clean_text(value, max_len: int = 120):
        if not isinstance(value, str):
            return ""
        cleaned = re.sub(r"\s+", " ", value).strip()
        if not cleaned:
            return ""
        if len(cleaned) <= max_len:
            return cleaned
        return cleaned[: max_len - 3].rstrip() + "..."

    @staticmethod
    def _clean_optional_text(value, max_len: int = 80):
        cleaned = RiskAssessAgent._clean_text(value, max_len=max_len)
        return cleaned or None

    @staticmethod
    def _build_canonical_display_map(avoid_terms):
        display_by_canonical = {}
        for ingredient in avoid_terms:
            display_name = RiskAssessAgent._clean_text(ingredient, max_len=80)
            if not display_name:
                continue
            canonical = get_canonical_ingredient(display_name, mode="input") or display_name
            canonical = canonical.casefold()
            if canonical not in display_by_canonical:
                display_by_canonical[canonical] = display_name
        return display_by_canonical

    @staticmethod
    def _normalize_canonical(value, allowed_canonicals):
        cleaned = RiskAssessAgent._clean_text(value, max_len=80)
        if not cleaned:
            return ""

        normalized = cleaned.casefold()
        if normalized in allowed_canonicals:
            return normalized

        canonical = get_canonical_ingredient(cleaned, mode="input")
        if canonical and canonical in allowed_canonicals:
            return canonical
        return ""

    @staticmethod
    def _menu_contains_span(menu_text: str, evidence_text: str) -> bool:
        menu_norm = norm(menu_text)
        evidence_norm = norm(evidence_text)
        if not menu_norm or not evidence_norm:
            return False
        return evidence_norm in menu_norm

    @staticmethod
    def _normalize_suspects(raw_suspects, allowed_canonicals, menu_text: str):
        if not isinstance(raw_suspects, list):
            return []

        legacy_evidence_map = {
            "common_recipe": "menu_prior",
            "uncertain": "weak_inference",
        }
        best_by_canonical = {}

        for raw in raw_suspects:
            if not isinstance(raw, dict):
                continue

            canonical = RiskAssessAgent._normalize_canonical(raw.get("canonical", ""), allowed_canonicals)
            if not canonical:
                continue

            evidence_type = str(raw.get("evidence_type", "none")).strip().casefold()
            evidence_type = legacy_evidence_map.get(evidence_type, evidence_type)
            if evidence_type not in SUSPECT_RANK:
                evidence_type = "none"
            if evidence_type == "none":
                continue

            evidence_text = RiskAssessAgent._clean_optional_text(raw.get("evidence_text", None), max_len=80)
            if evidence_text and not RiskAssessAgent._menu_contains_span(menu_text, evidence_text):
                evidence_text = None

            confidence = clamp_float(raw.get("confidence", 0.0), 0.0, 1.0, 0.0)
            if evidence_type == "direct" and not evidence_text:
                evidence_type = "weak_inference"
                confidence = min(confidence, WEAK_INFERENCE_MAX_CONF)
            elif evidence_type == "alias" and not evidence_text:
                evidence_type = "weak_inference"
                confidence = min(confidence, WEAK_INFERENCE_MAX_CONF)
            elif evidence_type == "weak_inference":
                confidence = min(confidence, WEAK_INFERENCE_MAX_CONF)

            suspect = RiskSuspect(
                canonical=canonical,
                evidence_type=evidence_type,
                evidence_text=evidence_text,
                reason=RiskAssessAgent._clean_text(raw.get("reason", ""), max_len=160),
                confidence=confidence,
            )

            current = best_by_canonical.get(canonical)
            if current is None:
                best_by_canonical[canonical] = suspect
                continue

            current_rank = SUSPECT_RANK.get(current.evidence_type, 0)
            new_rank = SUSPECT_RANK.get(suspect.evidence_type, 0)
            if new_rank > current_rank or (
                new_rank == current_rank and suspect.confidence > current.confidence
            ):
                best_by_canonical[canonical] = suspect

        return sorted(
            best_by_canonical.values(),
            key=lambda s: (SUSPECT_RANK.get(s.evidence_type, 0), s.confidence),
            reverse=True,
        )

    @staticmethod
    def _coerce_legacy_suspects(raw_item, allowed_canonicals, item_confidence: float):
        if not isinstance(raw_item, dict):
            return []

        legacy_suspects = []
        legacy_map = {
            "direct": "direct",
            "common_recipe": "menu_prior",
            "uncertain": "weak_inference",
        }

        raw_evidence = raw_item.get("avoid_evidence", [])
        if isinstance(raw_evidence, list):
            for evidence in raw_evidence:
                if not isinstance(evidence, dict):
                    continue
                canonical = RiskAssessAgent._normalize_canonical(evidence.get("ingredient", ""), allowed_canonicals)
                if not canonical:
                    continue
                legacy_suspects.append(
                    {
                        "canonical": canonical,
                        "evidence_type": legacy_map.get(
                            str(evidence.get("evidence_type", "uncertain")).strip().casefold(),
                            "weak_inference",
                        ),
                        "evidence_text": None,
                        "reason": evidence.get("note_ko", ""),
                        "confidence": item_confidence,
                    }
                )

        if legacy_suspects:
            return legacy_suspects

        raw_matched = raw_item.get("matched_avoid", [])
        if isinstance(raw_matched, list):
            for matched in raw_matched:
                canonical = RiskAssessAgent._normalize_canonical(matched, allowed_canonicals)
                if not canonical:
                    continue
                legacy_suspects.append(
                    {
                        "canonical": canonical,
                        "evidence_type": "weak_inference",
                        "evidence_text": None,
                        "reason": "",
                        "confidence": item_confidence,
                    }
                )
        return legacy_suspects

    @staticmethod
    def _build_compat_fields(suspects, display_by_canonical):
        matched_avoid = []
        avoid_evidence = []
        suspected_ingredients = []

        for suspect in suspects:
            display_name = display_by_canonical.get(suspect.canonical, suspect.canonical)
            if display_name not in matched_avoid:
                matched_avoid.append(display_name)
            if display_name not in suspected_ingredients:
                suspected_ingredients.append(display_name)
            avoid_evidence.append(
                AvoidEvidence(
                    ingredient=display_name,
                    canonical=suspect.canonical,
                    evidence_type=suspect.evidence_type,
                    evidence_text=suspect.evidence_text,
                    reason=suspect.reason,
                    confidence=suspect.confidence,
                )
            )

        return matched_avoid, avoid_evidence, suspected_ingredients

    def _run_batch(self, items, avoid) -> RiskAssessOutput:
        canonical_avoid = canonicalize_avoid_ingredients(avoid)
        canonical_avoid = [item.casefold() for item in canonical_avoid if isinstance(item, str) and item.strip()]
        canonical_avoid = list(dict.fromkeys(canonical_avoid))
        display_by_canonical = self._build_canonical_display_map(avoid)
        allowed_canonicals = set(canonical_avoid)

        prompt = f"""
You are a food-risk suspect generator for menu items.

INPUT
- Menu items (use ONLY these, do not add/remove): {json.dumps(items, ensure_ascii=False)}
- Allowed avoid canonicals (use ONLY these exact canonical strings in output): {json.dumps(canonical_avoid, ensure_ascii=False)}

TASK
For each menu item, identify only avoid-ingredient suspects that are plausibly related.
Do NOT output a final risk score.

Top-level confidence:
- confidence: overall confidence in your menu-level assessment, including empty suspect lists.

Each suspect must contain:
- canonical: one of the allowed canonical strings
- evidence_type: one of ["direct", "alias", "menu_prior", "weak_inference", "none"]
- evidence_text: exact menu substring when there is direct textual support; otherwise null
- reason: short explanation
- confidence: float 0.0~1.0

Evidence type definitions:
- direct: the menu string explicitly names the avoid ingredient itself
- alias: the menu string contains a strong alias, cut, or representative token for that canonical
- menu_prior: the dish identity is strongly associated with that canonical even without an explicit alias token
- weak_inference: the dish could contain that canonical, but recipe variation is large
- none: no meaningful relation; prefer omitting the suspect instead of using this

IMPORTANT
- Rule 1: Only judge canonicals from the allowed canonical list. Do not invent new canonicals.
- Rule 2: Every suspect must include canonical, evidence_type, evidence_text, and confidence.
- Rule 3: direct is allowed only when evidence_text is an exact substring of the menu string itself.
- Rule 4: If you only have general recipe intuition, use weak_inference only. Do not present it as certain.
- Rule 5: If you do not know, prefer suspects: [] or lower confidence. Do not force a match.
- If there is no plausible relation to the allowed canonical list, return suspects: [].
- Do NOT output unrelated canonicals.
- Do NOT output a final risk score.
- You must return exactly one output item for each input menu item.
- Keep the same order as the input list.
- Do not skip any item.
- Do not merge or split menu items.
- Prefer precision over recall: avoid false positives even if that means returning no match more often.
- If the evidence is weak or ambiguous, prefer weak_inference or no suspect over a false positive.
- For direct or alias, provide evidence_text as the exact menu substring that supports the suspect whenever possible.
- If you cannot point to an exact menu substring, do NOT use direct. Use menu_prior or weak_inference instead.

For "menu_name", copy the menu name EXACTLY from the input list (character-by-character). Do not translate or modify.

OUTPUT: Return ONLY valid JSON (no markdown) with this schema:
{{
  "items": [
    {{
      "menu_name": "string",
      "confidence": 0.0,
      "suspects": [
        {{
          "canonical": "string",
          "evidence_type": "direct|alias|menu_prior|weak_inference|none",
          "evidence_text": "string or null",
          "reason": "string",
          "confidence": 0.0
        }}
      ]
    }}
  ]
}}
""".strip()

        data = self._generate_assessment(prompt)

        # 변경 배경:
        # - 이전 한계점: 모델이 메뉴명을 약간 바꾸면 입력 메뉴와 매핑이 끊어질 수 있었다.
        # - 변경 내용: normalize key를 기준으로 원본 메뉴명을 복원하는 매핑 테이블을 사용한다.
        # 모델이 메뉴명을 약간 변형해도 원본 입력과 안전하게 매핑하기 위한 테이블
        allowed = {}
        for menu_name in items:
            key = norm(menu_name)
            if key and key not in allowed:
                allowed[key] = menu_name

        by_menu = {}

        for it in data["items"]:
            if not isinstance(it, dict):
                continue

            menu_raw = it.get("menu_name", it.get("menu", ""))
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

            conf = clamp_float(it.get("confidence", 0.0), 0.0, 1.0, 0.0)
            raw_suspects = it.get("suspects", [])
            suspects = self._normalize_suspects(raw_suspects, allowed_canonicals, menu)
            if not suspects:
                suspects = self._normalize_suspects(
                    self._coerce_legacy_suspects(it, allowed_canonicals, conf),
                    allowed_canonicals,
                    menu,
                )
            matched, avoid_evidence, suspected = self._build_compat_fields(suspects, display_by_canonical)

            by_menu[menu] = (
                RiskItem(
                    menu=menu,
                    risk=0,
                    confidence=conf,
                    suspected_ingredients=suspected,
                    suspects=suspects,
                    matched_avoid=matched,
                    avoid_evidence=avoid_evidence,
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
                    risk=0,
                    confidence=0.0,
                    suspected_ingredients=[],
                    suspects=[],
                    matched_avoid=[],
                    avoid_evidence=[],
                )
            )

        return RiskAssessOutput(items=out)
