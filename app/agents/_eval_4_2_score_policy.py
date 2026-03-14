from app.agents._0_contracts import ScoredItem, ScorePolicyInput, ScorePolicyOutput


class ScorePolicyAgent:
    """
    LLM 추론 결과(RiskItem)를 점수화하는 규칙 기반 Agent.
    점수 계산은 deterministic 하게 유지한다.
    """

    EVIDENCE_RANK = {"direct": 3, "common_recipe": 2, "uncertain": 1}
    BASE_RISK_BY_TYPE = {"direct": 85, "common_recipe": 60, "uncertain": 35}
    REASON_LABELS = {
        "direct": "direct evidence",
        "common_recipe": "common recipe",
        "uncertain": "inference",
    }

    EXTRA_RISK_PER_INGREDIENT = 6
    EXTRA_RISK_CAP = 12
    LOW_CONF_RISK_SCALE = 10
    LOW_CONF_REASON_THRESHOLD = 0.4

    NO_EVIDENCE_BASE = 8
    NO_EVIDENCE_UNCERTAINTY_SCALE = 12
    NO_EVIDENCE_PRIOR_CAP = 20

    def run(self, request: ScorePolicyInput) -> ScorePolicyOutput:
        scored = []

        for it in request.risk_items:
            # 1) 구조화 근거 기반 위험도(규칙)
            structured_risk = self._structured_risk(it)
            # 2) no-evidence fallback에서만 제한적으로 사용하는 LLM prior risk
            fallback_prior_risk = int(max(0, min(100, it.risk)))
            if not it.avoid_evidence:
                final_risk = self._no_evidence_risk(fallback_prior_risk, it.confidence)
            else:
                # 근거가 있으면 LLM prior는 무시하고, 최종 risk는 로컬 규칙으로 결정한다.
                final_risk = structured_risk

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

        scored.sort(key=lambda x: x.score, reverse=True)
        best = scored[0] if scored else None
        return ScorePolicyOutput(items=scored, best=best)

    @staticmethod
    def _structured_risk(item) -> int:
        per_ingredient = {}
        for ev in item.avoid_evidence:
            w = ScorePolicyAgent.EVIDENCE_RANK.get(ev.evidence_type, 1)
            current = per_ingredient.get(ev.ingredient, 0)
            # 같은 재료가 중복되면 가장 강한 근거만 반영
            per_ingredient[ev.ingredient] = max(current, w)

        if not per_ingredient:
            return ScorePolicyAgent._no_evidence_risk(0, item.confidence)

        strongest_evidence = ScorePolicyAgent._strongest_evidence(item)
        strongest_type = strongest_evidence.evidence_type if strongest_evidence else "uncertain"
        base_risk = ScorePolicyAgent.BASE_RISK_BY_TYPE[strongest_type]

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
            ScorePolicyAgent.REASON_LABELS["uncertain"],
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
    def _no_evidence_risk(prior_risk: int, confidence: float) -> int:
        # 근거가 전혀 없을 때도 낮은 confidence에서는 과도한 안전 점수를 주지 않는다.
        bounded_conf = ScorePolicyAgent._clamp_confidence(confidence)
        uncertainty_risk = int(
            round(
                ScorePolicyAgent.NO_EVIDENCE_BASE
                + (1.0 - bounded_conf) * ScorePolicyAgent.NO_EVIDENCE_UNCERTAINTY_SCALE
            )
        )
        bounded_prior = min(max(0, int(prior_risk)), ScorePolicyAgent.NO_EVIDENCE_PRIOR_CAP)
        return max(uncertainty_risk, bounded_prior)

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
