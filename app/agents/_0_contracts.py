# Agent간 데이터 schema 모음

from typing import Dict, List, Literal, Optional
from pydantic import BaseModel, Field


SupportedLang = Literal["ko", "en", "es"]


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
    include_bbox: bool = Field(False, description="OCR 결과에 bbox를 포함할지 여부")


class OCRLanguageCandidate(BaseModel):
    lang: str = Field("", description="OCR 후보 언어 코드")
    score: float = Field(0.0, description="자동 언어 감지 점수")
    line_count: int = Field(0, ge=0, description="후보 OCR에서 살아남은 라인 수")
    avg_confidence: float = Field(0.0, ge=0.0, le=1.0, description="후보 OCR 평균 confidence")
    script_ratio: float = Field(0.0, ge=0.0, le=1.0, description="해당 언어 스크립트 적합도")


class OCROutput(BaseModel):
    lines: List[OCRLine] = Field(default_factory=list, description="OCR 라인 결과")
    texts: List[str] = Field(default_factory=list, description="OCR 라인 텍스트만 추출한 결과")
    resolved_lang: str = Field("", description="실제 OCR에 사용된 언어 코드")
    lang_detection_source: str = Field("", description="언어 결정 방식(auto/manual)")
    lang_detection_candidates: List[OCRLanguageCandidate] = Field(
        default_factory=list,
        description="자동 감지 시 후보 언어 점수 목록",
    )


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


RiskEvidenceType = Literal["direct", "alias", "menu_prior", "weak_inference", "none"]


class AvoidEvidence(BaseModel):
    ingredient: str = Field(..., description="사용자 표시용 기피 재료명")
    canonical: str = Field("", description="내부 canonical 기피 재료명")
    evidence_type: RiskEvidenceType = Field(
        "none",
        description="근거 수준: direct/alias/menu_prior/weak_inference/none",
    )
    evidence_text: Optional[str] = Field(
        None,
        description="메뉴 문자열에서 직접 확인된 근거 토큰. 없으면 null",
    )
    reason: str = Field("", description="짧은 설명용 근거 메모")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="해당 근거의 신뢰도")


class RiskSuspect(BaseModel):
    canonical: str = Field(..., description="내부 canonical 기피 재료명")
    evidence_type: RiskEvidenceType = Field(
        "none",
        description="근거 수준: direct/alias/menu_prior/weak_inference/none",
    )
    evidence_text: Optional[str] = Field(
        None,
        description="메뉴 문자열에서 직접 확인된 근거 토큰. 없으면 null",
    )
    reason: str = Field("", description="짧은 설명용 근거 메모")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="해당 suspect의 신뢰도")


class RiskItem(BaseModel):
    menu: str = Field(..., description="입력 메뉴명과 동일한 원문")
    menu_original: str = Field("", description="추론/검증에 사용할 불변 원문 메뉴명")
    risk: int = Field(0, ge=0, le=100, description="deprecated placeholder; 최종 risk는 score policy에서 계산")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="메뉴 단위 평가 신뢰도")
    suspected_ingredients: List[str] = Field(
        default_factory=list,
        description="메뉴명/기피 추정에서 읽힌 재료 cue의 표시명 목록",
    )
    suspects: List[RiskSuspect] = Field(
        default_factory=list,
        description="기피 재료 관련 suspect 목록",
    )
    matched_avoid: List[str] = Field(
        default_factory=list,
        description="하위 호환용 표시 필드. suspect canonical을 사용자 표시명으로 변환한 결과",
    )
    avoid_evidence: List[AvoidEvidence] = Field(
        default_factory=list,
        description="하위 호환용 근거 목록. suspects를 기반으로 파생된다",
    )


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
    bbox_image_url: str = Field("", description="점수 상위 메뉴 bounding box 이미지 presigned URL")
    bbox_image_local_path: str = Field("", description="점수 상위 메뉴 bounding box 로컬 저장 경로")
    bbox_target_menus: List[str] = Field(default_factory=list, description="bounding box 대상으로 사용된 상위 메뉴명")
    timings_ms: Dict[str, int] = Field(default_factory=dict)
    output_lang: str = Field("ko", description="최종 사용자 표시/응답 언어")
    menu_country_code: str = Field("", description="OCR로 추정한 메뉴판 국가 코드")
    menu_ocr_lang: str = Field("", description="실제 OCR에 사용된 언어 코드")
    menu_ocr_lang_source: str = Field("", description="OCR 언어 결정 방식(auto/manual)")


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
