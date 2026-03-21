from app.agents._0_contracts import ScoredItem, ScorePolicyInput, ScorePolicyOutput
from app.utils.avoid_ingredient_synonyms import get_menu_evidence_catalog, normalize_ingredient_token


class ScorePolicyAgent:
    """
    LLM 추론 결과(RiskItem)를 uncertainty-aware 확률 점수로 변환한다.
    - evidence별 confidence를 calibration
    - ingredient 확률을 noisy-or로 합성
    - 정보 부족 uncertainty를 별도 위험 확률로 결합
    """

    EVIDENCE_RANK = {
        "direct": 4,
        "alias": 3,
        "menu_prior": 2,
        "weak_inference": 1,
        "none": 0,
    }
    EVIDENCE_PROB_FACTORS = {
        "direct": 1.00,
        "alias": 0.85,
        "menu_prior": 0.60,
        "weak_inference": 0.35,
        "none": 0.15,
    }
    REASON_LABELS = {
        "direct": "direct evidence",
        "alias": "alias hit",
        "menu_prior": "strong menu prior",
        "weak_inference": "weak inference",
        "none": "unsupported",
    }
    LOW_CONF_REASON_THRESHOLD = 0.25

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
        scored_pairs = []

        for it in request.risk_items:
            menu_profile = self._build_menu_profile(it)
            evidence_prob, ingredient_probs, top_evidence_by_key, ingredient_name_by_key = (
                self._risk_probability_from_evidence(it, menu_profile)
            )
            uncertainty_prob = self._uncertainty_probability(
                item=it,
                menu_profile=menu_profile,
                uncertainty_penalty=request.uncertainty_penalty,
                has_evidence=bool(ingredient_probs),
            )
            final_risk_prob = self._combine_probabilities([evidence_prob, uncertainty_prob])
            final_risk = int(max(0, min(100, round(final_risk_prob * 100.0))))
            final_score = int(max(0, min(100, 100 - final_risk)))

            display_confidence = self._relevant_confidence(it) if it.avoid_evidence else self._clamp_confidence(it.confidence)
            reason = self._build_reason_en(
                item=it,
                ingredient_probs=ingredient_probs,
                top_evidence_by_key=top_evidence_by_key,
                ingredient_name_by_key=ingredient_name_by_key,
                uncertainty_prob=uncertainty_prob,
            )

            scored_pairs.append(
                (
                    ScoredItem(
                        menu=self._original_menu_name(it),
                        menu_original=self._original_menu_name(it),
                        score=final_score,
                        risk=final_risk,
                        confidence=display_confidence,
                        matched_avoid=it.matched_avoid,
                        suspected_ingredients=it.suspected_ingredients,
                        reason=reason,
                    ),
                    it,
                )
            )

        scored_pairs.sort(key=lambda pair: self._sort_key(pair[0], pair[1]))
        scored = [item for item, _ in scored_pairs]
        best = scored[0] if scored else None
        return ScorePolicyOutput(items=scored, best=best)

    @classmethod
    def _risk_probability_from_evidence(cls, item, menu_profile: dict):
        per_ingredient_prob = {}
        top_evidence_by_key = {}
        ingredient_name_by_key = {}

        for ev in item.avoid_evidence:
            key = normalize_ingredient_token(ev.canonical or ev.ingredient)
            if not key:
                continue

            p = cls._calibrate_evidence_probability(ev, menu_profile)
            if p <= 0.0:
                continue

            current = per_ingredient_prob.get(key, 0.0)
            if p > current:
                per_ingredient_prob[key] = p
                top_evidence_by_key[key] = ev
                ingredient_name_by_key[key] = (ev.ingredient or ev.canonical or key)

        risk_prob = cls._combine_probabilities(per_ingredient_prob.values())
        return risk_prob, per_ingredient_prob, top_evidence_by_key, ingredient_name_by_key

    @classmethod
    def _calibrate_evidence_probability(cls, evidence, menu_profile: dict) -> float:
        evidence_type = str(evidence.evidence_type or "none")
        type_factor = cls.EVIDENCE_PROB_FACTORS.get(evidence_type, cls.EVIDENCE_PROB_FACTORS["none"])
        calibrated = cls._clamp_confidence(evidence.confidence) * type_factor

        specific_signal_count = int(menu_profile.get("specific_signal_count", 0))
        if specific_signal_count >= 2:
            calibrated *= 1.05
        elif specific_signal_count == 0:
            calibrated *= 0.90

        if menu_profile.get("is_generic_only"):
            if evidence_type in {"menu_prior", "weak_inference", "none"}:
                calibrated *= 0.85
        elif menu_profile.get("generic_hits"):
            if evidence_type in {"menu_prior", "weak_inference", "none"}:
                calibrated *= 0.93

        if evidence_type == "direct" and (evidence.evidence_text or "").strip():
            calibrated = max(calibrated, 0.65)
        if evidence_type == "none":
            calibrated = min(calibrated, 0.18)

        return max(0.0, min(0.99, float(calibrated)))

    @classmethod
    def _uncertainty_probability(
        cls,
        item,
        menu_profile: dict,
        uncertainty_penalty: int,
        has_evidence: bool,
    ) -> float:
        scale = max(0.0, min(1.0, float(uncertainty_penalty) / 100.0))
        specific_signal_count = int(menu_profile.get("specific_signal_count", 0))
        generic_hits_count = len(menu_profile.get("generic_hits", []))

        if has_evidence:
            relevant_conf = cls._relevant_confidence(item)
            uncertainty = (1.0 - relevant_conf) * (0.12 + 0.18 * scale)
            if menu_profile.get("is_generic_only"):
                uncertainty += 0.08
            elif generic_hits_count:
                uncertainty += min(0.06, generic_hits_count * 0.02)
            return max(0.0, min(0.35, uncertainty))

        menu_conf = cls._clamp_confidence(item.confidence)
        uncertainty = 0.10 + (1.0 - menu_conf) * (0.30 + 0.40 * scale)
        if specific_signal_count == 0:
            uncertainty += 0.08
        elif specific_signal_count >= 2:
            uncertainty -= 0.05
        else:
            uncertainty -= 0.02

        if menu_profile.get("is_generic_only"):
            uncertainty += 0.18
        elif generic_hits_count:
            uncertainty += min(0.08, generic_hits_count * 0.03)

        return max(0.08, min(0.90, uncertainty))

    @classmethod
    def _build_reason_en(
        cls,
        item,
        ingredient_probs: dict,
        top_evidence_by_key: dict,
        ingredient_name_by_key: dict,
        uncertainty_prob: float,
    ) -> str:
        if not ingredient_probs:
            if uncertainty_prob >= 0.45:
                return "Insufficient avoid-ingredient evidence (high uncertainty)"
            if uncertainty_prob >= 0.25:
                return "Insufficient avoid-ingredient evidence (moderate uncertainty)"
            return "Insufficient avoid-ingredient evidence"

        top_key = max(ingredient_probs, key=ingredient_probs.get)
        top_prob = ingredient_probs[top_key]
        top_ev = top_evidence_by_key.get(top_key)
        top_name = ingredient_name_by_key.get(top_key, top_key)
        top_type = str(top_ev.evidence_type) if top_ev is not None else "weak_inference"
        label = cls.REASON_LABELS.get(top_type, cls.REASON_LABELS["weak_inference"])

        unique_count = len(ingredient_probs)
        if unique_count == 1:
            reason = f"Caution: {top_name} ({label}, p={top_prob:.2f})"
        else:
            reason = f"Caution: {top_name} + {unique_count - 1} more ingredients (max p={top_prob:.2f})"

        if uncertainty_prob >= cls.LOW_CONF_REASON_THRESHOLD:
            return f"{reason} (uncertain)"
        return reason

    @staticmethod
    def _combine_probabilities(probabilities) -> float:
        survival = 1.0
        for p in probabilities:
            bounded = max(0.0, min(0.999, float(p)))
            survival *= (1.0 - bounded)
        return max(0.0, min(0.999, 1.0 - survival))

    @staticmethod
    def _strongest_evidence(item):
        if not item.avoid_evidence:
            return None
        return max(
            item.avoid_evidence,
            key=lambda ev: ScorePolicyAgent.EVIDENCE_RANK.get(ev.evidence_type, 0),
        )

    @classmethod
    def _relevant_confidence(cls, item) -> float:
        strongest = cls._strongest_evidence(item)
        if strongest is not None:
            return cls._clamp_confidence(strongest.confidence)
        return cls._clamp_confidence(item.confidence)

    @classmethod
    def _strongest_evidence_rank(cls, item) -> int:
        strongest = cls._strongest_evidence(item)
        if strongest is None:
            return 0
        return cls.EVIDENCE_RANK.get(strongest.evidence_type, 0)

    @staticmethod
    def _clamp_confidence(confidence: float) -> float:
        return max(0.0, min(1.0, float(confidence)))

    @classmethod
    def _sort_key(cls, scored_item, risk_item):
        has_avoid_evidence = 1 if risk_item.avoid_evidence else 0
        strongest_rank = cls._strongest_evidence_rank(risk_item)
        relevant_confidence = cls._relevant_confidence(risk_item)
        confidence_order = relevant_confidence if has_avoid_evidence else -relevant_confidence
        menu_key = normalize_ingredient_token(scored_item.menu_original or scored_item.menu)
        return (
            -scored_item.score,
            has_avoid_evidence,
            strongest_rank,
            confidence_order,
            menu_key,
        )

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
        menu_name = cls._original_menu_name(item)
        generic_hits = cls._collect_term_hits(menu_name, cls.GENERIC_TITLE_TERMS)
        catalog_hits = cls._collect_term_hits(menu_name, cls._GLOBAL_SPECIFIC_MENU_TERMS)
        dish_hits = cls._collect_term_hits(menu_name, cls.SPECIFIC_DISH_TERMS)
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

    @staticmethod
    def _original_menu_name(item) -> str:
        menu_original = getattr(item, "menu_original", "")
        if isinstance(menu_original, str) and menu_original.strip():
            return menu_original
        menu = getattr(item, "menu", "")
        return menu if isinstance(menu, str) else ""
