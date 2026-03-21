import os
import threading
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.agents._0_contracts import FinalResponse, TranslateOutput
from app.agents._0_orchestrator import ImageLoadError, MenuAgentOrchestrator
from app.agents._eval_2_ocr import OCRAgent
from app.clients.gemma_client import GemmaClient
from app.utils.env_loader import load_local_env


app = FastAPI(title="Menu AI API", version="0.1")
load_local_env()

# --- singletons ---
gemma = GemmaClient(
    api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("MODEL_ID", "gemma-3-4b-it"),
)
orchestrator = MenuAgentOrchestrator(gemma, uncertainty_penalty=40)


def _parse_preload_langs(raw_value: str) -> List[str]:
    if not raw_value:
        return list(OCRAgent.DEFAULT_AUTO_LANGS)
    out: List[str] = []
    for token in raw_value.split(","):
        value = token.strip().lower()
        if value:
            out.append(value)
    return out or list(OCRAgent.DEFAULT_AUTO_LANGS)


def _should_enable(value: str, default: bool = True) -> bool:
    raw = (value or "").strip().lower()
    if not raw:
        return default
    return raw not in {"0", "false", "no", "off"}


@app.on_event("startup")
def preload_ocr_engines():
    if not _should_enable(os.getenv("OCR_PRELOAD_ON_STARTUP", "1"), default=True):
        return

    preload_mode = (os.getenv("OCR_PRELOAD_MODE", "probe") or "probe").strip().lower()
    preload_probe = preload_mode in {"probe", "all"}
    preload_full = preload_mode in {"full", "all"}
    preload_langs = _parse_preload_langs(os.getenv("OCR_PRELOAD_LANGS", ""))

    def _worker():
        try:
            OCRAgent.warmup_shared_engines(
                langs=preload_langs,
                preload_probe=preload_probe,
                preload_full=preload_full,
            )
        except Exception:
            # warm-up failure should not block API startup
            return

    threading.Thread(target=_worker, daemon=True).start()


class RankRequest(BaseModel):
    image_url: str = Field(..., description="메뉴판 이미지 URL")
    presigned_url: str = Field(
        "",
        description="bbox 결과 이미지를 업로드할 presigned PUT URL",
    )
    avoid: List[str] = Field(default_factory=list, description="기피 재료 리스트")
    user_lang: Optional[str] = Field(None, description="사용자/응답 언어(ko/en/es)")
    menu_lang: Optional[str] = Field(
        None,
        description="메뉴판 OCR 언어(ko/en/es). 주어지면 자동 언어 탐지를 생략",
    )
    menu_country_code: Optional[str] = Field(
        None,
        description="메뉴판 OCR 언어 힌트. 모르면 AUTO",
    )
    # 하위 호환: 기존 필드가 오면 user_lang/menu_country_code로 대체 사용
    lang: Optional[str] = Field(None, description="(deprecated) user_lang 사용 권장")
    country_code: Optional[str] = Field(None, description="(deprecated) menu_country_code 사용 권장")


class TranslateRequest(BaseModel):
    texts: List[str] = Field(default_factory=list, description="번역할 텍스트 리스트")
    source_lang: str = Field("auto", description="원본 언어 코드")
    target_lang: str = Field("en", description="목표 언어 코드")


class AvoidIntakeRequest(BaseModel):
    user_text: str = Field(..., description="챗봇 사용자 입력 문장")
    lang: str = Field("ko", description="사용자 언어/응답 언어(ko/en/es)")


class AvoidIntakeResponse(BaseModel):
    candidates: List[str]
    confirm_question: str


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/rank", response_model=FinalResponse)
def rank(req: RankRequest):
    try:
        user_lang = ((req.user_lang if req.user_lang is not None else req.lang) or "ko")
        menu_lang = (req.menu_lang or "auto")
        menu_country_code = ((req.menu_country_code if req.menu_country_code is not None else req.country_code) or "AUTO")
        result = orchestrator.run(
            req.image_url,
            req.avoid,
            user_lang=user_lang,
            menu_lang=menu_lang,
            menu_country_code=menu_country_code,
            presigned_url=req.presigned_url,
        )
    except ImageLoadError as exc:
        raise HTTPException(
            status_code=400,
            detail={"code": "IMAGE_LOAD_FAILED", "message": str(exc)},
        ) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "RANK_PIPELINE_FAILED", "message": str(exc)},
        ) from exc

    if hasattr(result, "model_dump"):
        return result.model_dump()
    return result.dict()


@app.post("/translate", response_model=TranslateOutput)
def translate(req: TranslateRequest):
    try:
        out = orchestrator.translate_only(
            texts=req.texts,
            source_lang=req.source_lang,
            target_lang=req.target_lang,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "TRANSLATE_FAILED", "message": str(exc)},
        ) from exc

    if hasattr(out, "model_dump"):
        return out.model_dump()
    return out.dict()


@app.post("/avoid/intake", response_model=AvoidIntakeResponse)
def avoid_intake(req: AvoidIntakeRequest):
    try:
        out = orchestrator.intake_avoid(
            user_text=req.user_text,
            lang=req.lang,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail={"code": "AVOID_INTAKE_FAILED", "message": str(exc)},
        ) from exc

    intake = out.model_dump() if hasattr(out, "model_dump") else out
    return {
        "candidates": intake["candidates"] if isinstance(intake, dict) else [],
        "confirm_question": intake["confirm_question"] if isinstance(intake, dict) else "",
    }
