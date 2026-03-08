import argparse
import json
import os
import time

from app.agents.contracts import OCROptions
from app.agents.ocr_agent import OCRAgent
from app.agents.preprocess_agent import ImagePreprocessAgent
from app.utils.image_io import load_image


def build_parser():
    parser = argparse.ArgumentParser(description="전처리 -> OCR 성능 실험 스크립트")
    parser.add_argument("--image", default="menu_image/menu.png", help="입력 이미지 경로")
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_ocr.png",
        help="전처리 결과 저장 경로",
    )
    parser.add_argument(
        "--ocr-json-out",
        default="debug/ocr_result.json",
        help="OCR 결과 JSON 저장 경로",
    )
    parser.add_argument("--lang", default="korean", help="PaddleOCR language")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="OCR confidence 임계값")
    parser.add_argument(
        "--min-short-edge",
        type=int,
        default=900,
        help="짧은 변이 이 값보다 작을 때만 업스케일",
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

    preprocess = ImagePreprocessAgent(min_short_edge=args.min_short_edge)
    pre_data, _ = preprocess.run(raw_data, mime, save_path=args.preprocessed_out)
    t_pre = int((time.perf_counter() - t0) * 1000)

    ocr = OCRAgent(lang=args.lang)
    options = OCROptions(
        min_confidence=args.min_confidence,
    )

    try:
        t1 = time.perf_counter()
        out = ocr.run(pre_data, options=options)
        t_ocr = int((time.perf_counter() - t1) * 1000)
    except RuntimeError as exc:
        print("[ERROR] OCR 실행 실패:", str(exc))
        print("힌트: app/.venv 기준으로 `pip install paddleocr paddlepaddle` 후 다시 실행하세요.")
        return 1

    print("=== OCR Experiment Result ===")
    print(f"input image       : {args.image}")
    print(f"preprocessed image: {args.preprocessed_out}")
    print(f"line count        : {len(out.lines)}")
    print(f"preprocess ms     : {t_pre}")
    print(f"ocr ms            : {t_ocr}")
    print("")

    for idx, line in enumerate(out.lines, start=1):
        print(f"[{idx:03d}] conf={line.confidence:.3f} text={line.text}")

    ensure_parent(args.ocr_json_out)
    payload = {
        "input_image": args.image,
        "preprocessed_image": args.preprocessed_out,
        "timings_ms": {"preprocess": t_pre, "ocr": t_ocr},
        "options": options.model_dump() if hasattr(options, "model_dump") else options.dict(),
        "result": out.model_dump() if hasattr(out, "model_dump") else out.dict(),
    }
    with open(args.ocr_json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print(f"json saved         : {args.ocr_json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
