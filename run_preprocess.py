import argparse
import os

from app.agents._eval_1_img_preprocessor import ImagePreprocessAgent
from app.utils.image_io import load_image

try:
    import cv2
except Exception:
    cv2 = None

try:
    import numpy as np
except Exception:
    np = None


def build_parser():
    parser = argparse.ArgumentParser(description="전처리 전/후 이미지 비교")
    parser.add_argument("--image", default="menu_image/menu.png", help="입력 이미지")
    parser.add_argument("--out", default="debug/preprocessed_menu.png", help="전처리 결과 이미지")
    parser.add_argument("--compare-out", default="debug/preprocess_compare.png", help="전/후 비교 이미지")
    parser.add_argument("--min-short-edge", type=int, default=900, help="업스케일 기준")
    return parser


def ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def decode_image(data: bytes):
    if cv2 is None or np is None:
        return None
    buf = np.frombuffer(data, dtype=np.uint8)
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def labeled(img, text: str):
    out = img.copy()
    cv2.rectangle(out, (8, 8), (220, 40), (255, 255, 255), -1)
    cv2.putText(out, text, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def save_comparison(raw_data: bytes, pre_data: bytes, out_path: str) -> bool:
    if cv2 is None or np is None:
        return False

    raw = decode_image(raw_data)
    pre = decode_image(pre_data)
    if raw is None or pre is None:
        return False

    h = max(raw.shape[0], pre.shape[0])
    raw_w = int(round(raw.shape[1] * (h / raw.shape[0])))
    pre_w = int(round(pre.shape[1] * (h / pre.shape[0])))
    raw_r = cv2.resize(raw, (raw_w, h), interpolation=cv2.INTER_AREA)
    pre_r = cv2.resize(pre, (pre_w, h), interpolation=cv2.INTER_AREA)

    gap = np.full((h, 20, 3), 235, dtype=np.uint8)
    comp = np.hstack([labeled(raw_r, "BEFORE"), gap, labeled(pre_r, "AFTER")])

    ensure_parent(out_path)
    return bool(cv2.imwrite(out_path, comp))


def main():
    args = build_parser().parse_args()

    data, mime = load_image(args.image)
    agent = ImagePreprocessAgent(min_short_edge=args.min_short_edge)
    out_data, out_mime = agent.run(data, mime, save_path=args.out)

    compare_ok = save_comparison(data, out_data, args.compare_out)

    print("done")
    print("source :", args.image)
    print("saved  :", args.out)
    print("compare:", args.compare_out if compare_ok else "(failed)")
    print("mime   :", out_mime)
    print("bytes  :", len(out_data))


if __name__ == "__main__":
    main()
