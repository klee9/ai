import argparse
import json
import os
import time

from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.agents._eval_3_5_image_only_extractor import (
    ImageOnlyMenuExtractAgent,
    build_image_only_menu_extraction_prompt,
)
from app.clients.gemma_client import GemmaClient
from app.utils.image_io import load_image
from app.utils.menu_item_cleaner import clean_menu_candidates


def build_parser():
    parser = argparse.ArgumentParser(description="이미지 -> LLM 직접 메뉴 추출 실험 (OCR 미사용)")
    parser.add_argument("--image", default="menu_image/menu_korean.jpeg", help="입력 이미지 경로")
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_image_only_extract.png",
        help="전처리 이미지 저장 경로",
    )
    parser.add_argument(
        "--json-out",
        default="debug/image_only_menu_extract_result.json",
        help="실험 결과 JSON 저장 경로",
    )
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="이미지 전처리 없이 원본 이미지를 그대로 LLM에 전달",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="콘솔에 출력할 최대 개수",
    )
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="실행 프롬프트를 콘솔에 출력",
    )
    return parser


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = build_parser().parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY가 설정되어 있지 않습니다.")
        print("예: export GOOGLE_API_KEY='your-key'")
        return 1

    model_id = os.getenv("MODEL_ID", "gemma-3-4b-it")
    prompt = build_image_only_menu_extraction_prompt()

    if args.show_prompt:
        print("=== PROMPT ===")
        print(prompt)
        print("")

    t0 = time.perf_counter()
    raw_data, raw_mime = load_image(args.image)
    t_load = int((time.perf_counter() - t0) * 1000)

    image_data = raw_data
    image_mime = raw_mime
    t_pre = 0
    if not args.no_preprocess:
        t_pre_s = time.perf_counter()
        pre = ImagePreprocessAgent()
        image_data, image_mime = pre.run(raw_data, raw_mime, save_path=args.preprocessed_out)
        t_pre = int((time.perf_counter() - t_pre_s) * 1000)

    t_ext_s = time.perf_counter()
    gemma = GemmaClient(api_key=api_key, model=model_id)
    extractor = ImageOnlyMenuExtractAgent(gemma)
    try:
        extracted = extractor.run(image_data, image_mime)
    except Exception as exc:
        print("[ERROR] image-only extractor failed:", str(exc))
        return 1
    t_ext = int((time.perf_counter() - t_ext_s) * 1000)

    t_clean_s = time.perf_counter()
    clean_result = clean_menu_candidates(extracted.items)
    t_clean = int((time.perf_counter() - t_clean_s) * 1000)

    top_n = max(0, int(args.top))

    print("=== Image Only Menu Extract Result ===")
    print(f"image             : {args.image}")
    print(f"with preprocess   : {not args.no_preprocess}")
    if not args.no_preprocess:
        print(f"preprocessed      : {args.preprocessed_out}")
    print(f"raw item count    : {len(extracted.items)}")
    print(f"clean item count  : {len(clean_result.cleaned_items)}")
    print(f"timings(ms)       : load={t_load}, preprocess={t_pre}, extract={t_ext}, clean={t_clean}")
    print("")

    print("--- RAW EXTRACTED ITEMS ---")
    for idx, text in enumerate(extracted.items[:top_n], start=1):
        print(f"[RAW  {idx:03d}] {text}")
    if len(extracted.items) > top_n:
        print(f"... ({len(extracted.items) - top_n} more)")

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
        "input_image": args.image,
        "with_preprocess": bool(not args.no_preprocess),
        "preprocessed_image": args.preprocessed_out if not args.no_preprocess else "",
        "prompt": prompt,
        "raw_extracted_items": list(extracted.items),
        "clean_result": clean_result.to_dict(),
        "timings_ms": {
            "load": t_load,
            "preprocess": t_pre,
            "extract": t_ext,
            "clean": t_clean,
        },
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print(f"json saved        : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
