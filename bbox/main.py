import argparse
from ocr_bbox import GoogleVisionBBox

# 사용 예시
# python main.py --image images/test_1.jpg --targets "특상삼겹" "통항정살"
def main(img_path, targets):
    box_maker = GoogleVisionBBox()
    box_maker.predict(img_path)
    box_maker.draw_boxes(img_path, targets)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--targets", nargs="*")

    args = parser.parse_args()

    main(args.image, args.targets)