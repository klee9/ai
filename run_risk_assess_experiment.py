import argparse
import json
import os
import time
from typing import List

from app.agents._0_contracts import OCROptions, RiskAssessInput
from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.agents._eval_2_ocr import OCRAgent
from app.agents._eval_3_extractor import OCRMenuJudgeAgent
from app.agents._eval_4_1_risk_assessor import RiskAssessAgent
from app.clients.gemma_client import GemmaClient
from app.utils.image_io import load_image
from app.utils.menu_evidence_verifier import verify_risk_items
from app.utils.menu_item_cleaner import clean_menu_candidates


def build_parser():
    parser = argparse.ArgumentParser(description="menu items -> risk assessor -> local verify 실험")
    parser.add_argument("--image", default="", help="입력 이미지 경로. 비우면 메뉴 리스트 모드 사용")
    parser.add_argument(
        "--items",
        nargs="*",
        default=None,
        help="직접 넣을 메뉴 리스트. 예: --items '제육한상' '떡갈비한상'",
    )
    parser.add_argument(
        "--items-json",
        default="",
        help="메뉴 리스트 JSON 파일 경로. 예: ['제육한상', '떡갈비한상']",
    )
    parser.add_argument(
        "--avoid",
        nargs="*",
        default=["돼지고기"],
        help="기피 재료 목록. 예: --avoid 돼지고기 계란",
    )
    parser.add_argument(
        "--user-lang",
        default="ko",
        choices=["ko", "en", "cn"],
        help="출력 표시 언어",
    )
    parser.add_argument("--menu-country-code", default="KR", help="국가 코드(ISO-3166 alpha-2)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="OCR confidence 임계값")
    parser.add_argument(
        "--min-short-edge",
        type=int,
        default=900,
        help="짧은 변이 이 값보다 작을 때만 업스케일",
    )
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_risk_assess.png",
        help="전처리 이미지 저장 경로(--image 사용 시)",
    )
    parser.add_argument(
        "--json-out",
        default="debug/risk_assess_result.json",
        help="실험 결과 JSON 저장 경로",
    )
    parser.add_argument("--top", type=int, default=30, help="콘솔에 출력할 최대 개수")
    return parser


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_items_from_args(args) -> List[str]:
    if args.items:
        return [value for value in args.items if isinstance(value, str) and value.strip()]

    if args.items_json:
        with open(args.items_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [value for value in payload if isinstance(value, str) and value.strip()]
        raise ValueError("items-json must contain a JSON list of strings")

    return []


def extract_items_from_image(args) -> dict:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for image-based extraction mode")

    model_id = os.getenv("MODEL_ID", "gemma-3-4b-it")

    t0 = time.perf_counter()
    raw_data, mime = load_image(args.image)
    t_load = int((time.perf_counter() - t0) * 1000)

    t_pre_s = time.perf_counter()
    preprocess = ImagePreprocessAgent(min_short_edge=args.min_short_edge)
    pre_data, _ = preprocess.run(raw_data, mime, save_path=args.preprocessed_out)
    t_pre = int((time.perf_counter() - t_pre_s) * 1000)

    t_ocr_s = time.perf_counter()
    ocr = OCRAgent(menu_country_code=args.menu_country_code)
    ocr_opts = OCROptions(min_confidence=args.min_confidence)
    ocr_out = ocr.run(pre_data, options=ocr_opts)
    t_ocr = int((time.perf_counter() - t_ocr_s) * 1000)

    t_ext_s = time.perf_counter()
    gemma = GemmaClient(api_key=api_key, model=model_id)
    extractor = OCRMenuJudgeAgent(gemma)
    judged = extractor.run_lines_with_image(
        lines=ocr_out.lines,
        image_bytes=raw_data,
        image_mime=mime,
        use_image_context=True,
    )
    t_ext = int((time.perf_counter() - t_ext_s) * 1000)

    t_clean_s = time.perf_counter()
    clean_result = clean_menu_candidates(judged.menu_texts)
    t_clean = int((time.perf_counter() - t_clean_s) * 1000)

    return {
        "raw_candidates": list(judged.menu_texts),
        "cleaned_items": [item.cleaned_text for item in clean_result.kept],
        "timings_ms": {
            "image_load": t_load,
            "preprocess": t_pre,
            "ocr": t_ocr,
            "extractor": t_ext,
            "clean": t_clean,
        },
        "ocr_result": ocr_out.model_dump() if hasattr(ocr_out, "model_dump") else ocr_out.dict(),
        "extractor_result": judged.model_dump() if hasattr(judged, "model_dump") else judged.dict(),
        "clean_result": clean_result.to_dict(),
    }


def suspect_to_dict(suspect):
    if hasattr(suspect, "model_dump"):
        return suspect.model_dump()
    return suspect.dict()


def print_suspects(header: str, items, top_n: int):
    print(header)
    if not items:
        print("- 없음")
        print("")
        return

    for idx, item in enumerate(items[:top_n], start=1):
        print(f"[{idx:03d}] menu={item.menu} | conf={item.confidence:.2f}")
        if not item.suspects:
            print("      suspects=[]")
            continue
        for suspect in item.suspects:
            print(
                "      "
                f"canonical={suspect.canonical} | "
                f"type={suspect.evidence_type} | "
                f"evidence_text={suspect.evidence_text} | "
                f"conf={suspect.confidence:.2f} | "
                f"reason={suspect.reason or '-'}"
            )
    if len(items) > top_n:
        print(f"... ({len(items) - top_n} more)")
    print("")


def main():
    args = build_parser().parse_args()

    menu_items = load_items_from_args(args)
    extraction_meta = {
        "raw_candidates": [],
        "cleaned_items": [],
        "timings_ms": {},
    }

    if menu_items:
        extraction_meta["cleaned_items"] = menu_items
    elif args.image:
        try:
            extraction_meta = extract_items_from_image(args)
        except Exception as exc:
            print("[ERROR] image-based extraction failed:", str(exc))
            return 1
        menu_items = extraction_meta.get("cleaned_items", [])
    else:
        print("[ERROR] either --image or --items/--items-json is required")
        return 1

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY가 설정되어 있지 않습니다.")
        print("예: export GOOGLE_API_KEY='your-key'")
        return 1

    model_id = os.getenv("MODEL_ID", "gemma-3-4b-it")
    gemma = GemmaClient(api_key=api_key, model=model_id)
    risk_agent = RiskAssessAgent(gemma)

    t_risk_s = time.perf_counter()
    try:
        raw_risk = risk_agent.run(RiskAssessInput(items=menu_items, avoid=args.avoid))
    except Exception as exc:
        print("[ERROR] risk assessor failed:", str(exc))
        return 1
    t_risk = int((time.perf_counter() - t_risk_s) * 1000)

    t_verify_s = time.perf_counter()
    verified_items = verify_risk_items(raw_risk.items, avoid_terms=args.avoid, lang=args.user_lang)
    t_verify = int((time.perf_counter() - t_verify_s) * 1000)

    top_n = max(0, int(args.top))

    print("=== Risk Assess Experiment ===")
    print(f"input mode        : {'image' if args.image and not args.items and not args.items_json else 'menu_items'}")
    if args.image:
        print(f"image             : {args.image}")
    print(f"avoid             : {', '.join(args.avoid) if args.avoid else '-'}")
    print(f"menu item cnt     : {len(menu_items)}")
    if extraction_meta.get("timings_ms"):
        print(f"timings(ms)       : {extraction_meta['timings_ms']}, risk_assess={t_risk}, verify={t_verify}")
    else:
        print(f"timings(ms)       : risk_assess={t_risk}, verify={t_verify}")
    print("")

    print("--- MENU ITEMS ---")
    if not menu_items:
        print("- 없음")
    else:
        for idx, item in enumerate(menu_items[:top_n], start=1):
            print(f"[ITEM {idx:03d}] {item}")
        if len(menu_items) > top_n:
            print(f"... ({len(menu_items) - top_n} more)")
    print("")

    if extraction_meta.get("raw_candidates"):
        print("--- RAW EXTRACTED CANDIDATES ---")
        for idx, item in enumerate(extraction_meta["raw_candidates"][:top_n], start=1):
            print(f"[RAW  {idx:03d}] {item}")
        if len(extraction_meta["raw_candidates"]) > top_n:
            print(f"... ({len(extraction_meta['raw_candidates']) - top_n} more)")
        print("")

    print_suspects("--- RAW RISK ASSESSOR OUTPUT ---", raw_risk.items, top_n)
    print_suspects("--- VERIFIED OUTPUT ---", verified_items, top_n)

    ensure_parent(args.json_out)
    payload = {
        "input_mode": "image" if args.image and not args.items and not args.items_json else "menu_items",
        "image": args.image,
        "menu_items": menu_items,
        "avoid": list(args.avoid),
        "raw_risk_output": raw_risk.model_dump() if hasattr(raw_risk, "model_dump") else raw_risk.dict(),
        "verified_risk_items": [
            item.model_dump() if hasattr(item, "model_dump") else item.dict() for item in verified_items
        ],
        "timings_ms": {
            **(extraction_meta.get("timings_ms") or {}),
            "risk_assess": t_risk,
            "verify": t_verify,
        },
    }
    if extraction_meta.get("ocr_result") is not None:
        payload["ocr_result"] = extraction_meta["ocr_result"]
    if extraction_meta.get("extractor_result") is not None:
        payload["extractor_result"] = extraction_meta["extractor_result"]
    if extraction_meta.get("clean_result") is not None:
        payload["clean_result"] = extraction_meta["clean_result"]

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"json saved        : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
