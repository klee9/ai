import argparse
import json
import os
import time
from typing import List

from app.agents._0_contracts import OCROptions
from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.agents._eval_2_ocr import OCRAgent
from app.agents._eval_3_extractor import OCRMenuJudgeAgent
from app.clients.gemma_client import GemmaClient
from app.utils.image_io import load_image
from app.utils.menu_item_cleaner import clean_menu_candidates


def build_parser():
    parser = argparse.ArgumentParser(description="raw menu candidates -> local cleaner 검증 실험")
    parser.add_argument("--image", default="", help="입력 이미지 경로. 비우면 후보 리스트 모드 사용")
    parser.add_argument(
        "--candidates",
        nargs="*",
        default=None,
        help="직접 넣을 raw 후보 리스트. 예: --candidates '담아낸한상' '떡갈비 두쪽0.7'",
    )
    parser.add_argument(
        "--candidates-json",
        default="",
        help="raw 후보 리스트 JSON 파일 경로. 예: ['담아낸한상', '떡갈비 두쪽0.7']",
    )
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_menu_clean.png",
        help="전처리 이미지 저장 경로(--image 사용 시)",
    )
    parser.add_argument(
        "--json-out",
        default="debug/menu_clean_result.json",
        help="실험 결과 JSON 저장 경로",
    )
    parser.add_argument("--menu-country-code", default="KR", help="국가 코드(ISO-3166 alpha-2, 예: KR/US/JP)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="OCR confidence 임계값")
    parser.add_argument(
        "--min-short-edge",
        type=int,
        default=900,
        help="짧은 변이 이 값보다 작을 때만 업스케일",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="콘솔에 출력할 최대 개수",
    )
    return parser


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def load_candidates_from_args(args) -> List[str]:
    if args.candidates:
        return [value for value in args.candidates if isinstance(value, str)]

    if args.candidates_json:
        with open(args.candidates_json, "r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, list):
            return [value for value in payload if isinstance(value, str)]
        raise ValueError("candidates-json must contain a JSON list of strings")

    return []


def extract_candidates_from_image(args) -> dict:
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

    return {
        "raw_candidates": list(judged.menu_texts),
        "timings_ms": {
            "image_load": t_load,
            "preprocess": t_pre,
            "ocr": t_ocr,
            "extractor": t_ext,
        },
        "ocr_result": ocr_out.model_dump() if hasattr(ocr_out, "model_dump") else ocr_out.dict(),
        "extractor_result": judged.model_dump() if hasattr(judged, "model_dump") else judged.dict(),
    }


def main():
    args = build_parser().parse_args()

    raw_candidates = load_candidates_from_args(args)
    extraction_meta = {
        "raw_candidates": [],
        "timings_ms": {},
    }

    if raw_candidates:
        extraction_meta["raw_candidates"] = raw_candidates
    elif args.image:
        try:
            extraction_meta = extract_candidates_from_image(args)
        except Exception as exc:
            print("[ERROR] image-based candidate extraction failed:", str(exc))
            return 1
        raw_candidates = extraction_meta.get("raw_candidates", [])
    else:
        print("[ERROR] either --image or --candidates/--candidates-json is required")
        return 1

    t_clean_s = time.perf_counter()
    clean_result = clean_menu_candidates(raw_candidates)
    t_clean = int((time.perf_counter() - t_clean_s) * 1000)

    top_n = max(0, int(args.top))

    print("=== Menu Clean Result ===")
    print(f"input mode        : {'image' if args.image and not args.candidates and not args.candidates_json else 'raw_candidates'}")
    if args.image:
        print(f"image             : {args.image}")
    print(f"raw candidate cnt : {len(raw_candidates)}")
    print(f"cleaned item cnt  : {len(clean_result.cleaned_items)}")
    if extraction_meta.get("timings_ms"):
        print(f"timings(ms)       : {extraction_meta['timings_ms']}, clean={t_clean}")
    else:
        print(f"timings(ms)       : clean={t_clean}")
    print("")

    print("--- RAW CANDIDATES ---")
    for idx, text in enumerate(raw_candidates[:top_n], start=1):
        print(f"[RAW  {idx:03d}] {text}")
    if len(raw_candidates) > top_n:
        print(f"... ({len(raw_candidates) - top_n} more)")

    print("")
    print("--- CLEANED ITEMS ---")
    for idx, item in enumerate(clean_result.kept[:top_n], start=1):
        print(
            f"[KEEP {idx:03d}] raw={item.raw_text} | cleaned={item.cleaned_text} | "
            f"score={item.score:.2f} | reasons={','.join(item.reasons) if item.reasons else '-'}"
        )
    if len(clean_result.kept) > top_n:
        print(f"... ({len(clean_result.kept) - top_n} more)")

    print("")
    print("--- DROPPED ITEMS ---")
    if not clean_result.dropped:
        print("- none")
    else:
        for idx, item in enumerate(clean_result.dropped[:top_n], start=1):
            print(
                f"[DROP {idx:03d}] raw={item.raw_text} | cleaned={item.cleaned_text} | "
                f"score={item.score:.2f} | reasons={','.join(item.reasons) if item.reasons else '-'}"
            )
        if len(clean_result.dropped) > top_n:
            print(f"... ({len(clean_result.dropped) - top_n} more)")

    ensure_parent(args.json_out)
    payload = {
        "input_mode": "image" if args.image and not args.candidates and not args.candidates_json else "raw_candidates",
        "image": args.image,
        "raw_candidates": raw_candidates,
        "clean_result": clean_result.to_dict(),
        "timings_ms": {
            **(extraction_meta.get("timings_ms") or {}),
            "clean": t_clean,
        },
    }
    if extraction_meta.get("ocr_result") is not None:
        payload["ocr_result"] = extraction_meta["ocr_result"]
    if extraction_meta.get("extractor_result") is not None:
        payload["extractor_result"] = extraction_meta["extractor_result"]

    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print(f"json saved        : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
