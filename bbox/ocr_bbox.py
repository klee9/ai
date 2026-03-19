import os
import re
import requests
import base64
from PIL import Image, ImageDraw
from pathlib import Path

class GoogleVisionBBox:
    def __init__(self):
        self.api_key = "Google Vision API 키 입력"
        self.data = None
        self.boxes = []

    def predict(self, image_path, request_type: str = "DOCUMENT_TEXT_DETECTION"):
        with open(image_path, "rb") as image_file:
            content = base64.b64encode(image_file.read()).decode("utf-8")
        
        # Google Vision API로 보낼 요청 생성
        url = f"https://vision.googleapis.com/v1/images:annotate?key={self.api_key}"
        payload = {
            "requests": [
                {
                    "image": {"content": content},
                    "features": [{"type": request_type}]
                }
            ]
        }
        response = requests.post(url, json=payload)
        self.data = response.json()
    
    def draw_boxes(self, image_path: str, targets: list[str]):
        image = Image.open(image_path)
        path = Path(image_path)
        filename = path.name

        responses = self.data.get('responses', [])
        if not responses or 'fullTextAnnotation' not in responses[0]:
            print("탐지된 메뉴가 없거나 OCR이 실패했습니다.")
        else:
            annotation = responses[0]['fullTextAnnotation']
            for page in annotation['pages']:
                for block in page['blocks']:
                    for paragraph in block['paragraphs']:
                        # 한 문단에 있는 텍스트 병합 (글자 잘림 현상 방지)
                        menu = "".join([
                            "".join([symbol['text'] for symbol in word['symbols']])
                            for word in paragraph['words']
                        ])
                        menu = re.sub(r"\s+", "", menu)

                        # 바운딩 박스 좌표
                        box = paragraph['boundingBox']['vertices']

                        # 이미지에 바운딩 박스 그리기
                        for item in targets:
                            temp = item # 로깅 위함. 삭제해도 됨
                            item = re.sub(r"\s+", "", item)
                            if item.lower() in menu.lower():
                                self.boxes.append(box)
                                print(f'메뉴: {temp}, 위치: {box}')

            for box in self.boxes:
                points = [(p["x"], p["y"]) for p in box]

                ImageDraw.Draw(image).polygon(points, outline="green", width=3)
            
            save_dir = 'pred'
            os.makedirs(save_dir, exist_ok=True)

            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")

            image.save(f'{save_dir}/{filename}')