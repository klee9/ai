import argparse
import json
import os
import time

from app.agents.contracts import OCROptions
from app.agents.menu_text_judge_agent import OCRMenuJudgeAgent
from app.agents.ocr_agent import OCRAgent
from app.agents.preprocess_agent import ImagePreprocessAgent
from app.clients.gemma_client import GemmaClient
from app.utils.image_io import load_image


def build_parser():
    parser = argparse.ArgumentParser(description="이미지 전처리 -> OCR -> 메뉴 판독 실험")
    parser.add_argument("--image", default="menu_image/menu.png", help="입력 이미지 경로")
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_menu_cycle.png",
        help="전처리 이미지 저장 경로",
    )
    parser.add_argument(
        "--json-out",
        default="debug/menu_cycle_result.json",
        help="실험 결과 JSON 저장 경로",
    )
    parser.add_argument("--ocr-lang", default="korean", help="PaddleOCR 언어 코드")
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
        default=20,
        help="콘솔에 출력할 메뉴 후보 최대 개수",
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

    pre = ImagePreprocessAgent(min_short_edge=args.min_short_edge)
    pre_data, _ = pre.run(raw_data, mime, save_path=args.preprocessed_out)
    t_pre = int((time.perf_counter() - t0) * 1000)

    ocr = OCRAgent(lang=args.ocr_lang)
    ocr_opts = OCROptions(min_confidence=args.min_confidence)
    try:
        t1 = time.perf_counter()
        ocr_out = ocr.run(pre_data, options=ocr_opts)
        t_ocr = int((time.perf_counter() - t1) * 1000)
    except Exception as exc:
        print("[ERROR] OCR 실패:", str(exc))
        return 1

    gemma = GemmaClient(api_key=api_key, model=model_id)
    judge = OCRMenuJudgeAgent(gemma)
    try:
        t2 = time.perf_counter()
        judged = judge.run_lines_with_image(
            lines=ocr_out.lines,
            image_bytes=raw_data,
            image_mime=mime,
            use_image_context=True,
        )
        t_judge = int((time.perf_counter() - t2) * 1000)
    except Exception as exc:
        print("[ERROR] Gemma 메뉴 판독 실패:", str(exc))
        return 1

    menu_texts = judged.menu_texts

    print("=== Menu Cycle Result ===")
    print(f"image            : {args.image}")
    print(f"preprocessed     : {args.preprocessed_out}")
    print(f"ocr lines        : {len(ocr_out.lines)}")
    print(f"menu candidates  : {len(menu_texts)}")
    print(f"timings(ms)      : preprocess={t_pre}, ocr={t_ocr}, judge={t_judge}")
    print("")

    top_n = max(0, int(args.top))
    for idx, text in enumerate(menu_texts[:top_n], start=1):
        print(f"[{idx:03d}] {text}")

    ensure_parent(args.json_out)
    payload = {
        "input_image": args.image,
        "preprocessed_image": args.preprocessed_out,
        "timings_ms": {
            "preprocess": t_pre,
            "ocr": t_ocr,
            "menu_judge": t_judge,
        },
        "ocr_options": ocr_opts.model_dump() if hasattr(ocr_opts, "model_dump") else ocr_opts.dict(),
        "ocr_result": ocr_out.model_dump() if hasattr(ocr_out, "model_dump") else ocr_out.dict(),
        "menu_judge_result": judged.model_dump() if hasattr(judged, "model_dump") else judged.dict(),
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print(f"json saved        : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
