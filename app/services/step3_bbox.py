import io
import os
import re
import requests
import base64
from PIL import Image, ImageDraw
from pathlib import Path


class MenuBBox:
    def __init__(self, api_key: str | None = None):
        # API 키 환경 변수 처리
        resolved = api_key or os.getenv("GOOGLE_VISION_API_KEY")
        if not resolved:
            raise RuntimeError(
                "Google Vision API 키가 없습니다."
                ".env 파일에서 GOOGLE_VISION_API_KEY를 설정해 주세요."
            )
        self.api_key = resolved
        self.data = None
        self.boxes = []

    def predict(self, image, request_type: str = "DOCUMENT_TEXT_DETECTION"):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")  # or "JPEG"

        content = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
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
    
    def draw(self, image_url: str, targets: list[str]):
        res = requests.get(image_url)
        image = Image.open(io.BytesIO(res.content))

        # OCR한 후 바운딩 박스 예측
        self.predict(image, request_type="DOCUMENT_TEXT_DETECTION")

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

            if image.mode in ("RGBA", "P"):
                image = image.convert("RGB")
            
            
            buffer = io.BytesIO()
            image.save(buffer, format="JPEG")

            return buffer.getvalue()