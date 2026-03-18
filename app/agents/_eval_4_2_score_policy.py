from app.agents._0_contracts import ScoredItem, ScorePolicyInput, ScorePolicyOutput
from app.utils.avoid_ingredient_synonyms import get_menu_evidence_catalog, normalize_ingredient_token


class ScorePolicyAgent:
    """
    LLM 추론 결과(RiskItem)를 점수화하는 규칙 기반 Agent.
    점수 계산은 deterministic 하게 유지한다.
    """

    EVIDENCE_RANK = {
        "direct": 4,
        "alias": 3,
        "menu_prior": 2,
        "weak_inference": 1,
        "none": 0,
    }
    BASE_RISK_BY_TYPE = {
        "direct": 100,
        "alias": 90,
        "menu_prior": 50,
        "weak_inference": 12,
        "none": 0,
    }
    REASON_LABELS = {
        "direct": "direct evidence",
        "alias": "alias hit",
        "menu_prior": "strong menu prior",
        "weak_inference": "weak inference",
        "none": "no evidence",
    }

    EXTRA_RISK_PER_INGREDIENT = 6
    EXTRA_RISK_CAP = 12
    LOW_CONF_RISK_SCALE = 6
    LOW_CONF_REASON_THRESHOLD = 0.4

    NO_EVIDENCE_BASE = 0
    NO_EVIDENCE_UNCERTAINTY_SCALE = 36
    NO_EVIDENCE_SPECIFIC_CONF_FLOOR_WEAK = 0.25
    NO_EVIDENCE_SPECIFIC_CONF_FLOOR_STRONG = 0.35

    GENERIC_ONLY_BASE_RISK = 30
    GENERIC_EXTRA_TOKEN_RISK = 5
    GENERIC_SUFFIX_RISK = 6
    GENERIC_LOW_CONF_EXTRA_RISK = 8
    GENERIC_SHORT_TITLE_EXTRA_RISK = 4

    SPECIFICITY_WEAK_BONUS = 6
    SPECIFICITY_STRONG_BONUS = 12

    GENERIC_TITLE_TERMS = (
        "한상",
        "차림",
        "한상차림",
        "계절한상차림",
        "계절",
        "세트",
        "정식",
        "추천",
        "추천메뉴",
        "시그니처",
        "signature",
        "signature set",
        "special",
        "seasonal",
        "set",
        "combo",
        "chef",
        "chef special",
        "套餐",
        "套饭",
        "推荐",
        "招牌",
        "季节",
    )
    SPECIFIC_DISH_TERMS = (
        "찌개",
        "탕",
        "국",
        "볶음",
        "구이",
        "찜",
        "전골",
        "전",
        "밥",
        "국수",
        "면",
        "수제비",
        "덮밥",
        "비빔",
        "김치",
        "갈비",
        "샐러드",
        "버거",
        "샌드위치",
        "카레",
        "파스타",
        "피자",
        "soup",
        "stew",
        "rice",
        "noodle",
        "noodles",
        "pasta",
        "dumpling",
        "salad",
        "burger",
        "sandwich",
        "curry",
        "bbq",
        "grill",
        "hotpot",
        "饭",
        "面",
        "粉",
        "汤",
        "锅",
        "炒",
        "拌",
        "烤",
        "饺",
        "饼",
        "粥",
    )

    _MENU_EVIDENCE_CATALOG = get_menu_evidence_catalog()
    _GLOBAL_SPECIFIC_MENU_TERMS = tuple(
        dict.fromkeys(
            normalize_ingredient_token(term)
            for entry in _MENU_EVIDENCE_CATALOG.values()
            for section in ("direct", "strong", "prior")
            for term in entry.get(section, [])
            if normalize_ingredient_token(term)
        )
    )

    def run(self, request: ScorePolicyInput) -> ScorePolicyOutput:
        scored = []

        for it in request.risk_items:
            menu_profile = self._build_menu_profile(it)
            generic_penalty = self._compute_generic_penalty(it, menu_profile)
            specificity_bonus = self._compute_specificity_bonus(it, menu_profile)

            # 1) 구조화 근거 기반 위험도(규칙)
            structured_risk = self._structured_risk(it)
            if not it.avoid_evidence:
                final_risk = self._no_evidence_risk(
                    it.confidence,
                    menu_profile,
                    request.uncertainty_penalty,
                )
                final_risk = final_risk + generic_penalty - specificity_bonus
            else:
                final_risk = structured_risk
                final_risk = final_risk + generic_penalty

            final_risk = max(0, min(100, final_risk))
            final_score = int(max(0, min(100, 100 - final_risk)))

            scored.append(
                ScoredItem(
                    menu=it.menu,
                    menu_original=it.menu,
                    score=final_score,
                    risk=final_risk,
                    confidence=it.confidence,
                    matched_avoid=it.matched_avoid,
                    suspected_ingredients=it.suspected_ingredients,
                    # reason은 영어 canonical 문장으로 먼저 생성하고,
                    # 최종 언어(localization)는 오케스트레이터에서 처리한다.
                    reason=self._build_reason_en(it),
                )
            )

        scored.sort(key=lambda x: (x.score, x.confidence), reverse=True)
        best = scored[0] if scored else None
        return ScorePolicyOutput(items=scored, best=best)

    @staticmethod
    def _structured_risk(item) -> int:
        per_ingredient = {}
        for ev in item.avoid_evidence:
            w = ScorePolicyAgent.EVIDENCE_RANK.get(ev.evidence_type, 1)
            current = per_ingredient.get(ev.canonical or ev.ingredient, 0)
            # 같은 재료가 중복되면 가장 강한 근거만 반영
            per_ingredient[ev.canonical or ev.ingredient] = max(current, w)

        if not per_ingredient:
            return 0

        strongest_evidence = ScorePolicyAgent._strongest_evidence(item)
        strongest_type = strongest_evidence.evidence_type if strongest_evidence else "none"
        base_risk = ScorePolicyAgent.BASE_RISK_BY_TYPE.get(strongest_type, 0)

        unique_count = len(per_ingredient)
        additional_ingredient_risk = min(
            ScorePolicyAgent.EXTRA_RISK_CAP,
            max(0, unique_count - 1) * ScorePolicyAgent.EXTRA_RISK_PER_INGREDIENT,
        )

        bounded_conf = ScorePolicyAgent._clamp_confidence(item.confidence)
        low_conf_penalty = int(round((1.0 - bounded_conf) * ScorePolicyAgent.LOW_CONF_RISK_SCALE))

        return int(min(100, base_risk + additional_ingredient_risk + low_conf_penalty))

    @staticmethod
    def _build_reason_en(item) -> str:
        if not item.avoid_evidence:
            return "Insufficient avoid-ingredient evidence"

        top = ScorePolicyAgent._strongest_evidence(item)
        if top is None:
            return "Insufficient avoid-ingredient evidence"
        label = ScorePolicyAgent.REASON_LABELS.get(
            top.evidence_type,
            ScorePolicyAgent.REASON_LABELS["weak_inference"],
        )

        unique_count = len({ev.ingredient for ev in item.avoid_evidence})
        if unique_count == 1:
            reason = f"Caution: {top.ingredient} ({label})"
        else:
            reason = f"Caution: {top.ingredient} + {unique_count - 1} more ingredients"

        if item.confidence < ScorePolicyAgent.LOW_CONF_REASON_THRESHOLD:
            return f"{reason} (low confidence)"
        return reason

    @staticmethod
    def _no_evidence_risk(confidence: float, menu_profile: dict, uncertainty_penalty: int) -> int:
        # no-evidence는 안전 확정이 아니라 정보 부족 상태로 보고 보수 처리한다.
        bounded_conf = ScorePolicyAgent._clamp_confidence(confidence)
        specific_signal_count = int(menu_profile.get("specific_signal_count", 0))
        if specific_signal_count >= 2:
            bounded_conf = max(bounded_conf, ScorePolicyAgent.NO_EVIDENCE_SPECIFIC_CONF_FLOOR_STRONG)
        elif specific_signal_count == 1:
            bounded_conf = max(bounded_conf, ScorePolicyAgent.NO_EVIDENCE_SPECIFIC_CONF_FLOOR_WEAK)

        uncertainty_scale = max(0, min(100, int(uncertainty_penalty)))
        return int(
            round(
                ScorePolicyAgent.NO_EVIDENCE_BASE
                + (1.0 - bounded_conf) * uncertainty_scale
            )
        )

    @staticmethod
    def _strongest_evidence(item):
        if not item.avoid_evidence:
            return None
        return max(
            item.avoid_evidence,
            key=lambda ev: ScorePolicyAgent.EVIDENCE_RANK.get(ev.evidence_type, 1),
        )

    @staticmethod
    def _clamp_confidence(confidence: float) -> float:
        return max(0.0, min(1.0, float(confidence)))

    @staticmethod
    def _contains_term(menu_name: str, term: str) -> bool:
        menu_norm = normalize_ingredient_token(menu_name)
        term_norm = normalize_ingredient_token(term)
        if not menu_norm or not term_norm:
            return False
        if all("a" <= ch <= "z" or ch.isdigit() or ch in {" ", "_", "-"} for ch in term_norm):
            padded = f" {menu_norm} "
            return f" {term_norm} " in padded
        return term_norm in menu_norm

    @classmethod
    def _collect_term_hits(cls, menu_name: str, terms) -> list[str]:
        hits = []
        seen = set()
        for term in terms:
            term_norm = normalize_ingredient_token(term)
            if not term_norm or term_norm in seen:
                continue
            if cls._contains_term(menu_name, term):
                seen.add(term_norm)
                hits.append(term)
        return hits

    @classmethod
    def _build_menu_profile(cls, item) -> dict:
        generic_hits = cls._collect_term_hits(item.menu, cls.GENERIC_TITLE_TERMS)
        catalog_hits = cls._collect_term_hits(item.menu, cls._GLOBAL_SPECIFIC_MENU_TERMS)
        dish_hits = cls._collect_term_hits(item.menu, cls.SPECIFIC_DISH_TERMS)
        strong_evidence_hits = [
            ev for ev in item.avoid_evidence if ev.evidence_type in {"direct", "alias", "menu_prior"}
        ]

        specific_signal_count = 0
        if catalog_hits:
            specific_signal_count += 1
        if dish_hits:
            specific_signal_count += 1
        if strong_evidence_hits:
            specific_signal_count += 1

        return {
            "generic_hits": generic_hits,
            "catalog_hits": catalog_hits,
            "dish_hits": dish_hits,
            "specific_signal_count": specific_signal_count,
            "is_generic_only": bool(generic_hits) and specific_signal_count == 0,
        }

    @classmethod
    def _compute_generic_penalty(cls, item, menu_profile: dict) -> int:
        generic_hits = menu_profile.get("generic_hits", [])
        if not generic_hits:
            return 0

        if menu_profile.get("is_generic_only"):
            penalty = cls.GENERIC_ONLY_BASE_RISK + max(0, len(generic_hits) - 1) * cls.GENERIC_EXTRA_TOKEN_RISK
            if cls._clamp_confidence(item.confidence) < 0.35:
                penalty += cls.GENERIC_LOW_CONF_EXTRA_RISK
            menu_norm = normalize_ingredient_token(item.menu)
            if menu_norm and len(menu_norm) <= 8:
                penalty += cls.GENERIC_SHORT_TITLE_EXTRA_RISK
            return min(45, penalty)

        return min(12, cls.GENERIC_SUFFIX_RISK + max(0, len(generic_hits) - 1) * 2)

    @classmethod
    def _compute_specificity_bonus(cls, item, menu_profile: dict) -> int:
        specific_signal_count = int(menu_profile.get("specific_signal_count", 0))
        if specific_signal_count >= 2:
            return cls.SPECIFICITY_STRONG_BONUS
        if specific_signal_count == 1:
            return cls.SPECIFICITY_WEAK_BONUS
        return 0
