from typing import Any, Dict, List

from app.agents._0_contracts import RiskAssessInput, ScorePolicyInput
from app.agents._eval_4_1_risk_assessor import RiskAssessAgent
from app.agents._eval_4_2_score_policy import ScorePolicyAgent
from app.clients.gemma_client import GemmaClient


class MenuRanker:
    """
    하위 호환용 래퍼.
    기존 rank(items, avoid) 시그니처를 유지하면서
    내부 구현은 Agent(RiskAssessAgent + ScorePolicyAgent)로 위임한다.
    """

    def __init__(self, gemma: GemmaClient, uncertainty_penalty: int = 40):
        self.uncertainty_penalty = uncertainty_penalty
        self.risk_agent = RiskAssessAgent(gemma)
        self.score_agent = ScorePolicyAgent()

    def rank(self, items: List[str], avoid: List[str]) -> Dict[str, Any]:
        risk_output = self.risk_agent.run(RiskAssessInput(items=items, avoid=avoid))
        score_output = self.score_agent.run(
            ScorePolicyInput(
                risk_items=risk_output.items,
                uncertainty_penalty=self.uncertainty_penalty,
            )
        )

        if hasattr(score_output, "model_dump"):
            return score_output.model_dump()
        return score_output.dict()
