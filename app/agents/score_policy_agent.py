from app.agents._0_contracts import ScoredItem, ScorePolicyInput, ScorePolicyOutput


class ScorePolicyAgent:
    """
    LLM 추론 결과(RiskItem)를 점수화하는 규칙 기반 Agent.
    점수 계산은 deterministic 하게 유지한다.
    """

    def run(self, request: ScorePolicyInput) -> ScorePolicyOutput:
        scored = []

        for it in request.risk_items:
            # 1) 구조화 근거 기반 위험도(규칙)
            structured_risk = self._structured_risk(it)
            # 2) 모델이 준 위험도(prior)
            model_prior_risk = int(max(0, min(100, it.risk)))
            # 변경 배경:
            # - 이전 한계점: no-evidence를 일괄 low-risk(<=20) 처리해 false-safe가 크게 발생했다.
            # - 변경 내용: assessment failure는 prior 유지, 일반 no-evidence는 불확실성 기반 보수 위험도로 계산한다.
            # 3) no-match면 "무조건 안전"으로 처리하지 않고 불확실성 기반 보수 위험도를 사용
            if not it.avoid_evidence:
                if self._is_assessment_failure(it.why_ko):
                    final_risk = model_prior_risk
                else:
                    final_risk = self._no_evidence_risk(model_prior_risk, it.confidence)
            else:
                final_risk = max(structured_risk, model_prior_risk)

            base_score = 100 - final_risk
            # confidence가 낮을수록 추가 감점
            final_score = base_score - (1.0 - it.confidence) * request.uncertainty_penalty
            final_score = int(max(0, min(100, round(final_score))))

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
        weights = {"direct": 45, "common_recipe": 30, "uncertain": 15}
        per_ingredient = {}
        for ev in item.avoid_evidence:
            w = weights.get(ev.evidence_type, 15)
            current = per_ingredient.get(ev.ingredient, 0)
            # 같은 재료가 중복되면 가장 강한 근거만 반영
            per_ingredient[ev.ingredient] = max(current, w)

        evidence_risk = sum(per_ingredient.values())
        # 근거는 있는데 설명문이 비어 있으면 품질 페널티 부여
        explanation_penalty = 10 if per_ingredient and not (item.why_ko or "").strip() else 0
        return int(min(100, evidence_risk + explanation_penalty))

    @staticmethod
    def _build_reason_en(item) -> str:
        if not item.avoid_evidence:
            # Keep explicit fallback text when orchestrator marks risk assess failure.
            why = (item.why_ko or "").strip()
            if ScorePolicyAgent._is_assessment_failure(why):
                return "Risk assessment failed (conservative fallback)"
            return "Insufficient avoid-ingredient evidence"

        rank = {"direct": 3, "common_recipe": 2, "uncertain": 1}
        top = sorted(
            item.avoid_evidence,
            key=lambda ev: rank.get(ev.evidence_type, 1),
            reverse=True,
        )[0]
        labels = {"direct": "direct evidence", "common_recipe": "common recipe", "uncertain": "inference"}
        label = labels.get(top.evidence_type, labels["uncertain"])

        unique_count = len({ev.ingredient for ev in item.avoid_evidence})
        if unique_count == 1:
            reason = f"Caution: {top.ingredient} ({label})"
        else:
            reason = f"Caution: {top.ingredient} + {unique_count - 1} more ingredients"

        if item.confidence < 0.4:
            return f"{reason} (low confidence)"
        return reason

    @staticmethod
    def _no_evidence_risk(prior_risk: int, confidence: float) -> int:
        # 변경 배경:
        # - 이전 한계점: evidence가 없어도 높은 safety score가 나와 추천 안정성이 떨어졌다.
        # - 변경 내용: confidence가 낮을수록 risk floor를 올리고 prior upper bound를 함께 반영한다.
        # 근거가 없더라도 낮은 confidence에서는 보수적으로 위험을 올린다.
        bounded_conf = max(0.0, min(1.0, float(confidence)))
        uncertainty_risk = int(round(45 + (1.0 - bounded_conf) * 25))
        bounded_prior = min(max(0, int(prior_risk)), 70)
        return max(uncertainty_risk, bounded_prior)

    @staticmethod
    def _is_assessment_failure(why: str) -> bool:
        msg = (why or "").strip().lower()
        if not msg:
            return False
        return (
            "판단 실패" in msg
            or "assessment failed" in msg
            or "风险评估失败" in msg
        )
