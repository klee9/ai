import time
from typing import List, Optional

from app.agents._0_contracts import (
    AvoidIntakeInput,
    FinalResponse,
    RiskAssessInput,
    RiskItem,
    TranslateInput,
    TranslateOutput,
    ScoredItem,
    ScorePolicyInput,
)
from app.agents._chat_1_avoid_taker import AvoidIntakeAgent
from app.agents.extract_agent import MenuExtractAgent
from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.agents._eval_4_1_risk_assessor import RiskAssessAgent
from app.agents._eval_4_2_score_policy import ScorePolicyAgent
from app.agents._0_translate_agent import TranslateAgent
from app.clients.gemma_client import GemmaClient
from app.utils.image_io import load_image


class MenuAgentOrchestrator:
    """
    Agent 실행 순서를 통제하는 얇은 오케스트레이터.
    Agent 실행 순서(Preprocess -> Extract -> RiskAssess -> ScorePolicy)를 담당한다.
    """

    def __init__(self, gemma: GemmaClient, uncertainty_penalty: int = 40, max_risk_retries: int = 1):
        self.gemma = gemma
        self.uncertainty_penalty = uncertainty_penalty
        self.max_risk_retries = max(0, int(max_risk_retries))
        self.extract_agent = MenuExtractAgent(gemma)
        self.preprocess_agent = ImagePreprocessAgent()
        self.risk_assess_agent = RiskAssessAgent(gemma)
        self.score_policy_agent = ScorePolicyAgent()
        self.avoid_intake_agent = AvoidIntakeAgent(gemma)
        self.translate_agent = TranslateAgent(gemma)

    def run(
        self,
        image_url: str,
        avoid: List[str],
        user_lang: str = "ko",
        menu_country_code: str = "KR",
    ) -> FinalResponse:
        # 허용 언어 외 값은 기본 한국어로 처리
        lang = user_lang if user_lang in {"ko", "en", "cn"} else "ko"
        _ = (menu_country_code or "KR").strip().upper()
        t_total = time.perf_counter()
        timings_ms = {}

        # 1) Extract Agent
        t_image = time.perf_counter()
        try:
            data, mime = load_image(image_url)
        except Exception as exc:
            # 로컬 파일/URL 로드 실패는 API 레이어에서 400으로 매핑됨
            raise ImageLoadError(f"failed to load image from source: {image_url}") from exc
        timings_ms["image_load"] = self._elapsed_ms(t_image)

        t_preprocess = time.perf_counter()
        # Gemma 추출 전, 문서 정렬/대비/노이즈를 보정해 OCR 친화적인 입력으로 정규화한다.
        data, mime = self.preprocess_agent.run(data, mime)
        timings_ms["preprocess"] = self._elapsed_ms(t_preprocess)

        t_extract = time.perf_counter()
        img_part = self.gemma.image_part_from_bytes(data, mime)
        extracted = self.extract_agent.run(img_part)
        timings_ms["extract"] = self._elapsed_ms(t_extract)

        # 메뉴가 비어 있으면 후속 단계를 건너뛰고 바로 반환
        if not extracted.items:
            timings_ms["risk_assess"] = 0
            timings_ms["score_policy"] = 0
            timings_ms["total"] = self._elapsed_ms(t_total)
            return FinalResponse(items_extracted=[], items=[], best=None, timings_ms=timings_ms)

        # 2) RiskAssess Agent
        t_risk = time.perf_counter()
        risk_output = None
        # LLM 파싱 실패 대비 재시도
        for _ in range(self.max_risk_retries + 1):
            try:
                risk_output = self.risk_assess_agent.run(
                    RiskAssessInput(items=extracted.items, avoid=avoid)
                )
                break
            except Exception:
                continue
        timings_ms["risk_assess"] = self._elapsed_ms(t_risk)

        # 3) ScorePolicy Agent
        t_score = time.perf_counter()
        if risk_output is None:
            # Risk 단계가 끝내 실패하면 보수적 점수로 폴백
            score_output = self._fallback_score(extracted.items, lang=lang)
        else:
            score_output = self.score_policy_agent.run(
                ScorePolicyInput(
                    risk_items=risk_output.items,
                    uncertainty_penalty=self.uncertainty_penalty,
                    lang=lang,
                )
            )
        timings_ms["score_policy"] = self._elapsed_ms(t_score)

        scored_items: List[ScoredItem] = []
        for it in score_output.items:
            scored_items.append(it)

        # menu는 사용자 언어 표시용으로 로컬라이즈하고, menu_original에는 원문을 유지한다.
        self._localize_item_menus(scored_items, lang)
        # reason은 ScorePolicy에서 영어로 생성되므로, 최종 언어로 로컬라이즈한다.
        self._localize_item_reasons(scored_items, lang)

        # best는 정렬된 리스트의 첫 항목을 사용해 reason 번역 결과와 일치시킨다.
        best = scored_items[0] if scored_items else None
        timings_ms["total"] = self._elapsed_ms(t_total)

        return FinalResponse(
            items_extracted=extracted.items,
            items=scored_items,
            best=best,
            timings_ms=timings_ms,
        )

    def _fallback_score(self, items: List[str], lang: str = "ko"):
        # 모든 메뉴를 최악 위험(100)으로 처리해 안전 우선 정책을 적용
        fallback_risk_items = [
            RiskItem(
                menu=menu,
                risk=100,
                confidence=0.0,
                suspected_ingredients=[],
                matched_avoid=[],
                avoid_evidence=[],
            )
            for menu in items
        ]
        return self.score_policy_agent.run(
            ScorePolicyInput(
                risk_items=fallback_risk_items,
                uncertainty_penalty=self.uncertainty_penalty,
                lang=lang,
            )
        )

    @staticmethod
    def _pick_best(raw_best: object, scored_items: List[ScoredItem]) -> Optional[ScoredItem]:
        if isinstance(raw_best, ScoredItem):
            return raw_best

        if scored_items:
            return scored_items[0]
        return None

    @staticmethod
    def _elapsed_ms(start_time: float) -> int:
        # perf_counter 기반 경과시간(ms) 계산
        return int(max(0, round((time.perf_counter() - start_time) * 1000)))

    def _localize_item_menus(self, items: List[ScoredItem], lang: str) -> None:
        target_lang = lang if lang in {"ko", "en", "cn"} else "en"
        if not items:
            return

        for item in items:
            if not item.menu_original:
                item.menu_original = item.menu

        if target_lang == "en":
            return

        translate_candidates = []
        seen_menus = set()
        for item in items:
            original_menu = (item.menu_original or item.menu or "").strip()
            if not original_menu or original_menu in seen_menus:
                continue
            seen_menus.add(original_menu)
            translate_candidates.append(original_menu)

        if not translate_candidates:
            return

        translated_map = {}
        try:
            translated = self.translate_only(
                texts=translate_candidates,
                source_lang="auto",
                target_lang=target_lang,
            )
            for idx, src in enumerate(translate_candidates):
                if idx >= len(translated.items):
                    break
                translated_text = (translated.items[idx].translated or "").strip()
                if translated_text:
                    translated_map[src] = translated_text
        except Exception:
            return

        for item in items:
            original_menu = (item.menu_original or item.menu or "").strip()
            if original_menu in translated_map:
                item.menu = translated_map[original_menu]

    def _localize_item_reasons(self, items: List[ScoredItem], lang: str) -> None:
        target_lang = lang if lang in {"ko", "en", "cn"} else "en"
        if not items:
            return

        no_match_reason = {
            "ko": "기피 재료 근거 부족",
            "en": "Insufficient avoid-ingredient evidence",
            "cn": "缺乏忌口成分依据",
        }[target_lang]
        failure_reason = {
            "ko": "위험도 판단 실패(보수적 처리)",
            "en": "Risk assessment failed (conservative fallback)",
            "cn": "风险评估失败（保守处理）",
        }[target_lang]

        # 변경 배경:
        # - 이전 한계점: 메뉴별 reason 번역을 개별 호출해 메뉴 수만큼 LLM 호출이 발생했다.
        # - 변경 내용: unique reason만 수집해 배치 1회 호출 후 항목별로 재적용한다.
        translate_candidates = []
        seen_reasons = set()

        for item in items:
            reason_en = (item.reason or "").strip()
            if not reason_en:
                item.reason = no_match_reason
                continue

            # no-match는 번역 결과와 무관하게 고정 문구로 강제해 일관성을 유지한다.
            if not item.matched_avoid:
                if "risk assessment failed" in reason_en.lower():
                    item.reason = failure_reason
                else:
                    item.reason = no_match_reason
                continue

            if target_lang == "en":
                continue

            if reason_en in seen_reasons:
                continue
            seen_reasons.add(reason_en)
            translate_candidates.append(reason_en)

        if target_lang == "en" or not translate_candidates:
            return

        translated_map = {}
        try:
            translated = self.translate_only(
                texts=translate_candidates,
                source_lang="en",
                target_lang=target_lang,
            )
            # 변경 배경:
            # - 이전 한계점: 개별 호출에서는 동일 문장 번역 품질이 항목마다 흔들릴 수 있었다.
            # - 변경 내용: 배치 결과를 source->translated 맵으로 고정해 같은 reason에 같은 번역을 적용한다.
            for idx, src in enumerate(translate_candidates):
                if idx >= len(translated.items):
                    break
                translated_text = (translated.items[idx].translated or "").strip()
                if translated_text:
                    translated_map[src] = translated_text
        except Exception:
            # 번역 실패 시 영어 reason 유지
            return

        for item in items:
            reason_en = (item.reason or "").strip()
            if not item.matched_avoid:
                continue
            if reason_en in translated_map:
                item.reason = translated_map[reason_en]

    def translate_only(
        self,
        texts: List[str],
        source_lang: str = "auto",
        target_lang: str = "en",
    ) -> TranslateOutput:
        return self.translate_agent.run(
            TranslateInput(
                texts=texts,
                source_lang=source_lang,
                target_lang=target_lang,
            )
        )

    def intake_avoid(
        self,
        user_text: str,
        lang: str = "ko",
    ):
        lang = lang if lang in {"ko", "en", "cn"} else "ko"
        return self.avoid_intake_agent.run(AvoidIntakeInput(user_text=user_text, lang=lang))


class ImageLoadError(RuntimeError):
    pass
