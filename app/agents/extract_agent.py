from google.genai import types

from app.agents._0_contracts import ExtractOutput
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import extract_first_json_object, normalize_list


class MenuExtractAgent:
    def __init__(self, gemma: GemmaClient):
        self.gemma = gemma

    def run(self, image_part: types.Part) -> ExtractOutput:
        prompt = """
Extract ONLY standalone menu item names from the image.

Rules:
- Do NOT include prices.
- Do NOT include options/sizes/add-ons/toppings (e.g., items under "Make your own", "Extra", "+$").
- Output ONLY valid JSON in this format:
{ "items": ["Menu 1", "Menu 2", "..."] }
""".strip()

        # 이미지 + 지시 프롬프트를 함께 넣어 "메뉴명만" 추출한다.
        text = self.gemma.generate_text([image_part, prompt], max_output_tokens=900)
        # LLM이 코드블록/설명문을 섞어도 첫 JSON 객체만 안전하게 꺼낸다.
        data = extract_first_json_object(text)

        # 1차 응답이 JSON 형식을 못 지킨 경우, 동일 지시로 1회 재시도한다.
        if data is None:
            retry_prompt = (
                f"{prompt}\n\n"
                "Return ONLY one JSON object. No markdown, no extra text."
            )
            retry_text = self.gemma.generate_text([image_part, retry_prompt], max_output_tokens=900)
            data = extract_first_json_object(retry_text)

        data = data or {}
        # 중복/공백 정리를 통해 후속 Agent 입력을 안정화한다.
        items = normalize_list(data.get("items", []))
        return ExtractOutput(items=items)
