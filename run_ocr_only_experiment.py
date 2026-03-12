import argparse
import json
import os
import time

from app.agents._0_contracts import OCROptions
from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.agents._eval_2_ocr import OCRAgent
from app.utils.image_io import load_image


def build_parser():
    parser = argparse.ArgumentParser(description="OCR-only 실험 스크립트")
    parser.add_argument("--image", default="menu_image/menu.png", help="입력 이미지 경로")
    parser.add_argument(
        "--ocr-json-out",
        default="debug/ocr_only_result.json",
        help="OCR 결과 JSON 저장 경로",
    )
    parser.add_argument("--menu-country-code", default="KR", help="메뉴판 언어 국가 코드(ISO-3166 alpha-2)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="OCR confidence 임계값")
    parser.add_argument("--with-preprocess", action="store_true", help="전처리 후 OCR 실행")
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_ocr_only.png",
        help="전처리 이미지 저장 경로(--with-preprocess일 때 사용)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=30,
        help="콘솔에 출력할 OCR 라인 최대 개수",
    )
    return parser


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def main():
    args = build_parser().parse_args()

    t0 = time.perf_counter()
    raw_data, mime = load_image(args.image)
    t_load = int((time.perf_counter() - t0) * 1000)

    data_for_ocr = raw_data
    t_pre = 0
    if args.with_preprocess:
        t1 = time.perf_counter()
        pre = ImagePreprocessAgent()
        data_for_ocr, _ = pre.run(raw_data, mime, save_path=args.preprocessed_out)
        t_pre = int((time.perf_counter() - t1) * 1000)

    ocr = OCRAgent(menu_country_code=args.menu_country_code)
    opts = OCROptions(min_confidence=args.min_confidence)

    try:
        t2 = time.perf_counter()
        out = ocr.run(data_for_ocr, options=opts)
        t_ocr = int((time.perf_counter() - t2) * 1000)
    except Exception as exc:
        print("[ERROR] OCR 실행 실패:", str(exc))
        return 1

    print("=== OCR Only Result ===")
    print(f"image            : {args.image}")
    print(f"with preprocess  : {args.with_preprocess}")
    if args.with_preprocess:
        print(f"preprocessed     : {args.preprocessed_out}")
    print(f"resolved ocr lang: {ocr.lang}")
    print(f"line count       : {len(out.lines)}")
    print(f"timings(ms)      : load={t_load}, preprocess={t_pre}, ocr={t_ocr}")
    print("")

    top_n = max(0, int(args.top))
    for idx, line in enumerate(out.lines[:top_n], start=1):
        print(f"[{idx:03d}] conf={line.confidence:.3f} text={line.text}")
    if len(out.lines) > top_n:
        print(f"... ({len(out.lines) - top_n} more)")

    ensure_parent(args.ocr_json_out)
    payload = {
        "input_image": args.image,
        "with_preprocess": bool(args.with_preprocess),
        "preprocessed_image": args.preprocessed_out if args.with_preprocess else "",
        "menu_country_code": args.menu_country_code,
        "ocr_lang_resolved": ocr.lang,
        "timings_ms": {"load": t_load, "preprocess": t_pre, "ocr": t_ocr},
        "options": opts.model_dump() if hasattr(opts, "model_dump") else opts.dict(),
        "result": out.model_dump() if hasattr(out, "model_dump") else out.dict(),
    }
    with open(args.ocr_json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print(f"json saved       : {args.ocr_json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
