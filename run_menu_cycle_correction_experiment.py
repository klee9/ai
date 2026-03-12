import argparse
import json
import os
import time
from typing import Dict, List, Optional

from app.agents._0_contracts import OCRLine, OCROptions
from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.agents._eval_2_ocr import OCRAgent
from app.clients.gemma_client import GemmaClient
from app.utils.image_io import load_image
from app.utils.parsing import extract_first_json_object


ALLOWED_LABELS = {
    "menu_item",
    "category_header",
    "description",
    "price",
    "option",
    "temperature",
    "size_volume",
    "promo",
    "store_info",
    "other",
}


def build_parser():
    parser = argparse.ArgumentParser(description="이미지 전처리 -> OCR -> (이미지 기반 보정) 메뉴 판독 실험")
    parser.add_argument("--image", default="menu_image/menu.png", help="입력 이미지 경로")
    parser.add_argument(
        "--preprocessed-out",
        default="debug/preprocessed_for_menu_cycle_corrected.png",
        help="전처리 이미지 저장 경로",
    )
    parser.add_argument(
        "--json-out",
        default="debug/menu_cycle_corrected_result.json",
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
        default=20,
        help="콘솔에 출력할 메뉴 후보 최대 개수",
    )
    return parser


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def norm(s: str) -> str:
    return " ".join((s or "").split()).strip().casefold()


def normalize_lines(lines: List[OCRLine]) -> List[OCRLine]:
    out: List[OCRLine] = []
    seen = set()
    for line in lines:
        if not isinstance(line, OCRLine):
            continue
        text = " ".join((line.text or "").split()).strip()
        if not text:
            continue
        key = norm(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(OCRLine(text=text, confidence=line.confidence, bbox=line.bbox))
    return out


def to_index(v) -> Optional[int]:
    try:
        if v is None:
            return None
        return int(v)
    except Exception:
        return None


def build_correction_prompt(payload: List[Dict]) -> str:
    return f"""
You are given:
1) A menu image.
2) OCR lines extracted from that image.

Task:
- Classify each OCR line into exactly one label.
- For `menu_item` lines only, provide `corrected_menu` by reading the image and correcting OCR mistakes.

Hard definition:
- `menu_item`: a final, standalone product name that can be directly ordered (dish or drink).
- Category/section names, options/add-ons, temperature, size/volume, price, quantity, description, promotion, and store info are NOT `menu_item`.

Allowed labels:
- menu_item
- category_header
- description
- price
- option
- temperature
- size_volume
- promo
- store_info
- other

Rules:
- Keep `text` exactly as OCR input.
- Use the same `index` from input for each output item.
- `corrected_menu` must be non-empty only when label is `menu_item`.
- `corrected_menu` should be the most accurate visible menu name from the image (fix OCR typos/splits when possible).
- If uncertain, keep `corrected_menu` equal to the original OCR text for that line.

OCR text bundle (JSON array):
{payload}

Return JSON only:
{{
  "line_labels": [
    {{
      "index": 0,
      "text": "...",
      "label": "menu_item|category_header|description|price|option|temperature|size_volume|promo|store_info|other",
      "corrected_menu": "..."
    }}
  ]
}}
""".strip()


def parse_line_labels(data: dict, source_lines: List[OCRLine]) -> Dict[int, Dict[str, str]]:
    out: Dict[int, Dict[str, str]] = {}
    raw_lines = data.get("line_labels", [])
    if not isinstance(raw_lines, list):
        return out

    source_text_to_idx = {norm(line.text): idx for idx, line in enumerate(source_lines)}

    for item in raw_lines:
        if not isinstance(item, dict):
            continue

        idx = to_index(item.get("index"))
        text = " ".join(str(item.get("text", "")).split()).strip()
        label = " ".join(str(item.get("label", "other")).split()).strip().casefold()
        corrected = " ".join(str(item.get("corrected_menu", "")).split()).strip()

        if label not in ALLOWED_LABELS:
            label = "other"

        if idx is None and text:
            idx = source_text_to_idx.get(norm(text))
        if idx is None or idx < 0 or idx >= len(source_lines):
            continue

        out[idx] = {"label": label, "corrected_menu": corrected}

    return out


def judge_with_image_correction(
    gemma: GemmaClient,
    lines: List[OCRLine],
    image_bytes: bytes,
    image_mime: str,
) -> dict:
    source_lines = normalize_lines(lines)
    if not source_lines:
        return {"items": [], "menu_texts": [], "corrected_menu_texts": []}

    payload = [{"index": idx, "text": line.text} for idx, line in enumerate(source_lines)]
    prompt = build_correction_prompt(payload)
    img_part = gemma.image_part_from_bytes(image_bytes, image_mime)

    raw = gemma.generate_text([prompt, img_part], max_output_tokens=3500)
    data = extract_first_json_object(raw)
    if data is None:
        retry_raw = gemma.generate_text(
            [f"{prompt}\nReturn ONLY one JSON object.", img_part],
            max_output_tokens=3500,
        )
        data = extract_first_json_object(retry_raw)

    label_map = parse_line_labels(data or {}, source_lines)

    items = []
    menu_texts = []
    corrected_menu_texts = []
    seen_corrected = set()

    for idx, line in enumerate(source_lines):
        parsed = label_map.get(idx, {})
        label = parsed.get("label", "other")
        corrected_menu = parsed.get("corrected_menu", "")
        is_menu = label == "menu_item"

        if is_menu:
            menu_texts.append(line.text)
            corrected = corrected_menu or line.text
            key = norm(corrected)
            if key and key not in seen_corrected:
                seen_corrected.add(key)
                corrected_menu_texts.append(corrected)
        else:
            corrected = ""

        items.append(
            {
                "index": idx,
                "text": line.text,
                "label": label,
                "is_menu": is_menu,
                "corrected_menu": corrected,
            }
        )

    return {
        "items": items,
        "menu_texts": menu_texts,
        "corrected_menu_texts": corrected_menu_texts,
    }


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

    ocr = OCRAgent(menu_country_code=args.menu_country_code)
    ocr_opts = OCROptions(min_confidence=args.min_confidence)
    try:
        t1 = time.perf_counter()
        ocr_out = ocr.run(pre_data, options=ocr_opts)
        t_ocr = int((time.perf_counter() - t1) * 1000)
    except Exception as exc:
        print("[ERROR] OCR 실패:", str(exc))
        return 1

    gemma = GemmaClient(api_key=api_key, model=model_id)
    try:
        t2 = time.perf_counter()
        judged = judge_with_image_correction(
            gemma=gemma,
            lines=ocr_out.lines,
            image_bytes=raw_data,
            image_mime=mime,
        )
        t_judge = int((time.perf_counter() - t2) * 1000)
    except Exception as exc:
        print("[ERROR] Gemma 메뉴 보정 판독 실패:", str(exc))
        return 1

    menu_texts = judged.get("menu_texts", [])
    corrected_menu_texts = judged.get("corrected_menu_texts", [])

    print("=== Menu Cycle Corrected Result ===")
    print(f"image                 : {args.image}")
    print(f"preprocessed          : {args.preprocessed_out}")
    print(f"ocr lines             : {len(ocr_out.lines)}")
    print(f"menu candidates(raw)  : {len(menu_texts)}")
    print(f"menu candidates(fixed): {len(corrected_menu_texts)}")
    print(f"timings(ms)           : preprocess={t_pre}, ocr={t_ocr}, judge={t_judge}")
    print("")

    top_n = max(0, int(args.top))
    print("--- RAW MENU TEXTS (OCR) ---")
    for idx, text in enumerate(menu_texts[:top_n], start=1):
        print(f"[RAW  {idx:03d}] {text}")
    if len(menu_texts) > top_n:
        print(f"... ({len(menu_texts) - top_n} more)")

    print("")
    print("--- CORRECTED MENU TEXTS ---")
    for idx, text in enumerate(corrected_menu_texts[:top_n], start=1):
        print(f"[FIXD {idx:03d}] {text}")
    if len(corrected_menu_texts) > top_n:
        print(f"... ({len(corrected_menu_texts) - top_n} more)")

    ensure_parent(args.json_out)
    payload = {
        "input_image": args.image,
        "preprocessed_image": args.preprocessed_out,
        "timings_ms": {
            "preprocess": t_pre,
            "ocr": t_ocr,
            "menu_judge_corrected": t_judge,
        },
        "ocr_options": ocr_opts.model_dump() if hasattr(ocr_opts, "model_dump") else ocr_opts.dict(),
        "ocr_result": ocr_out.model_dump() if hasattr(ocr_out, "model_dump") else ocr_out.dict(),
        "menu_judge_corrected_result": judged,
    }
    with open(args.json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("")
    print(f"json saved             : {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
