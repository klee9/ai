import argparse
import json
import os
import time

from app.agents._0_contracts import OCROptions
from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.agents._eval_2_ocr import OCRAgent
from app.agents._eval_3_extractor import OCRMenuJudgeAgent
from app.clients.gemma_client import GemmaClient
from app.utils.image_io import load_image


def build_parser():
    parser = argparse.ArgumentParser(description="전처리 -> OCR -> Extractor 로컬 실험")
    parser.add_argument("--image", default="menu_image/menu.png", help="입력 이미지 경로")
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_ocr_experiment.png",
        help="전처리 이미지 저장 경로",
    )
    parser.add_argument(
        "--json-out",
        default="debug/ocr_experiment_result.json",
        help="실험 결과 JSON 저장 경로",
    )
    parser.add_argument("--menu-country-code", default="KR", help="메뉴판 언어 국가 코드(ISO-3166 alpha-2)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="OCR confidence 임계값")
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="콘솔에 출력할 최대 라인 수",
    )
    parser.add_argument(
        "--no-image-context",
        action="store_true",
        help="Extractor에서 이미지 컨텍스트 비활성화",
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

    t0 = time.perf_counter()
    raw_data, mime = load_image(args.image)

    t_pre_s = time.perf_counter()
    preprocess = ImagePreprocessAgent()
    pre_data, _ = preprocess.run(raw_data, mime, save_path=args.preprocessed_out)
    t_pre = int((time.perf_counter() - t_pre_s) * 1000)

    t_ocr_s = time.perf_counter()
    ocr = OCRAgent(menu_country_code=args.menu_country_code)
    ocr_opts = OCROptions(min_confidence=args.min_confidence)
    try:
        ocr_out = ocr.run(pre_data, options=ocr_opts)
    except Exception as exc:
        print("[ERROR] OCR 실행 실패:", str(exc))
        return 1
    t_ocr = int((time.perf_counter() - t_ocr_s) * 1000)

    t_ext_s = time.perf_counter()
    gemma = GemmaClient(api_key=api_key, model=model_id)
    extractor = OCRMenuJudgeAgent(gemma)
    try:
        judged = extractor.run_lines_with_image(
            lines=ocr_out.lines,
            image_bytes=raw_data,
            image_mime=mime,
            use_image_context=(not args.no_image_context),
        )
    except Exception as exc:
        print("[ERROR] Extractor 실행 실패:", str(exc))
        return 1
    t_ext = int((time.perf_counter() - t_ext_s) * 1000)
    t_total = int((time.perf_counter() - t0) * 1000)

    label_groups = {}
    for item in judged.items:
        label_groups.setdefault(item.label, []).append(item.text)

    top_n = max(0, int(args.top))
    print("=== OCR Experiment Result ===")
    print(f"image              : {args.image}")
    print(f"preprocessed       : {args.preprocessed_out}")
    print(f"menu_country_code  : {args.menu_country_code}")
    print(f"resolved_ocr_lang  : {ocr.lang}")
    print(f"image_context      : {not args.no_image_context}")
    print(f"ocr_lines          : {len(ocr_out.lines)}")
    print(f"menu_texts         : {len(judged.menu_texts)}")
    print(f"timings(ms)        : preprocess={t_pre}, ocr={t_ocr}, extractor={t_ext}, total={t_total}")
    print("")

    print("--- OCR LINES ---")
    for idx, line in enumerate(ocr_out.lines[:top_n], start=1):
        print(f"[OCR  {idx:03d}] conf={line.confidence:.3f} text={line.text}")
    if len(ocr_out.lines) > top_n:
        print(f"... ({len(ocr_out.lines) - top_n} more)")

    print("")
    print("--- MENU TEXTS ---")
    for idx, text in enumerate(judged.menu_texts[:top_n], start=1):
        print(f"[MENU {idx:03d}] {text}")
    if len(judged.menu_texts) > top_n:
        print(f"... ({len(judged.menu_texts) - top_n} more)")

    if label_groups:
        print("")
        print("--- LABEL BREAKDOWN ---")
        for label in sorted(label_groups.keys()):
            texts = label_groups[label]
            print(f"{label:16}: {len(texts)}")
            for idx, text in enumerate(texts[:top_n], start=1):
                print(f"  [{idx:03d}] {text}")
            if len(texts) > top_n:
                print(f"  ... ({len(texts) - top_n} more)")

    ensure_parent(args.json_out)
    payload = {
        "input_image": args.image,
        "preprocessed_image": args.preprocessed_out,
        "menu_country_code": args.menu_country_code,
        "ocr_lang_resolved": ocr.lang,
        "timings_ms": {
            "preprocess": t_pre,
            "ocr": t_ocr,
            "extractor": t_ext,
            "total": t_total,
        },
        "ocr_options": ocr_opts.model_dump() if hasattr(ocr_opts, "model_dump") else ocr_opts.dict(),
        "ocr_result": ocr_out.model_dump() if hasattr(ocr_out, "model_dump") else ocr_out.dict(),
        "extractor_result": judged.model_dump() if hasattr(judged, "model_dump") else judged.dict(),
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print(f"json saved         : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
