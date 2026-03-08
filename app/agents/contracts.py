from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


SupportedLang = Literal["ko", "en", "cn"]


class ExtractInput(BaseModel):
    image_url: str = Field(..., description="메뉴판 이미지 URL")


class ExtractOutput(BaseModel):
    items: List[str] = Field(default_factory=list, description="추출된 메뉴명 리스트")


class OCRLine(BaseModel):
    text: str = Field("", description="OCR로 인식한 원문 텍스트")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="OCR 신뢰도")
    bbox: List[List[float]] = Field(
        default_factory=list,
        description="텍스트 박스 4점 좌표([[x1,y1],...,[x4,y4]])",
    )


class OCROptions(BaseModel):
    min_confidence: float = Field(0.5, ge=0.0, le=1.0, description="최소 신뢰도 필터")


class OCROutput(BaseModel):
    lines: List[OCRLine] = Field(default_factory=list, description="OCR 라인 결과")
    texts: List[str] = Field(default_factory=list, description="OCR 라인 텍스트만 추출한 결과")


class OCRTextLabel(BaseModel):
    text: str = Field("", description="OCR 원문 텍스트")
    label: str = Field(
        "other",
        description="분류 라벨(menu_item/section_header/description/price/option/other)",
    )
    is_menu: bool = Field(False, description="음식 메뉴명 여부")


class OCRMenuJudgeOutput(BaseModel):
    items: List[OCRTextLabel] = Field(default_factory=list, description="텍스트별 메뉴명 판정")
    menu_texts: List[str] = Field(default_factory=list, description="메뉴명으로 판정된 텍스트")


class AvoidEvidence(BaseModel):
    ingredient: str = Field(..., description="avoid 리스트에 포함된 재료명")
    evidence_type: Literal["direct", "common_recipe", "uncertain"] = Field(
        "uncertain",
        description="근거 수준: direct/common_recipe/uncertain",
    )
    note_ko: str = Field("", description="짧은 한국어 근거 메모")


class RiskItem(BaseModel):
    menu: str = Field(..., description="입력 메뉴명과 동일한 원문")
    risk: int = Field(50, ge=0, le=100, description="0~100, 높을수록 위험")
    confidence: float = Field(0.5, ge=0.0, le=1.0, description="0.0~1.0, 높을수록 확신")
    suspected_ingredients: List[str] = Field(
        default_factory=list, description="추정 핵심 성분(최대 3 권장)"
    )
    matched_avoid: List[str] = Field(
        default_factory=list, description="avoid 리스트와 일치하는 항목만"
    )
    avoid_evidence: List[AvoidEvidence] = Field(
        default_factory=list,
        description="avoid 성분별 근거 목록",
    )
    why_ko: str = Field("", description="짧은 한국어 근거")


class RiskAssessInput(BaseModel):
    items: List[str] = Field(default_factory=list, description="메뉴명 리스트")
    avoid: List[str] = Field(default_factory=list, description="기피 재료 리스트")


class RiskAssessOutput(BaseModel):
    items: List[RiskItem] = Field(default_factory=list, description="메뉴별 위험도 평가 결과")


class ScoredItem(BaseModel):
    menu: str
    menu_original: str = Field("", description="메뉴판 원문 메뉴명(번역 전)")
    score: int = Field(0, ge=0, le=100, description="최종 점수(높을수록 안전)")
    risk: int = Field(50, ge=0, le=100)
    confidence: float = Field(0.5, ge=0.0, le=1.0)
    matched_avoid: List[str] = Field(default_factory=list)
    suspected_ingredients: List[str] = Field(default_factory=list)
    reason: str = ""


class ScorePolicyInput(BaseModel):
    risk_items: List[RiskItem] = Field(default_factory=list)
    uncertainty_penalty: int = Field(40, ge=0, le=100)
    lang: SupportedLang = "ko"


class ScorePolicyOutput(BaseModel):
    items: List[ScoredItem] = Field(default_factory=list, description="score 내림차순 정렬")
    best: Optional[ScoredItem] = Field(None, description="최고 점수 메뉴")


class FinalResponse(BaseModel):
    items_extracted: List[str] = Field(default_factory=list)
    items: List[ScoredItem] = Field(default_factory=list)
    best: Optional[ScoredItem] = None
    timings_ms: Dict[str, int] = Field(default_factory=dict)


class TranslateInput(BaseModel):
    texts: List[str] = Field(default_factory=list, description="번역할 문자열 리스트")
    source_lang: str = Field("auto", description="원본 언어 코드. auto 허용")
    target_lang: str = Field("en", description="목표 언어 코드")


class TranslateItem(BaseModel):
    original: str = Field(..., description="원본 문자열")
    translated: str = Field(..., description="번역 문자열")
    confidence: float = Field(1.0, ge=0.0, le=1.0, description="번역 확신도(추정)")
    provider: str = Field("identity", description="번역 제공자 식별자")


class TranslateOutput(BaseModel):
    items: List[TranslateItem] = Field(default_factory=list)
    source_lang: str = "auto"
    target_lang: str = "en"


class AvoidIntakeInput(BaseModel):
    user_text: str = Field(..., description="자유형 사용자 입력 문장")
    lang: SupportedLang = "ko"


class AvoidIntakeOutput(BaseModel):
    candidates: List[str] = Field(default_factory=list, description="추출된 기피 재료 후보")
    confirm_question: str = Field("", description="사용자 확인 질문")
    confirm_question_ko: str = Field("", description="사용자 확인 질문")
