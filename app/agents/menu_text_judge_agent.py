from typing import List, Optional

from app.agents.contracts import OCRLine, OCRMenuJudgeOutput, OCRTextLabel
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import extract_first_json_object, normalize_list


class OCRMenuJudgeAgent:
    """OCR 텍스트 묶음 + 메뉴 이미지를 Gemma에 넣어 라벨링하는 최소 Agent."""

    def __init__(self, gemma: GemmaClient):
        self.gemma = gemma

    def run(self, texts: List[str]) -> OCRMenuJudgeOutput:
        candidates = normalize_list(texts, limit=300)
        lines = [OCRLine(text=t, confidence=1.0, bbox=[]) for t in candidates]
        return self.run_lines(lines)

    def run_lines(self, lines: List[OCRLine]) -> OCRMenuJudgeOutput:
        return self.run_lines_with_image(lines=lines, image_bytes=None, image_mime=None)

    def run_lines_with_image(
        self,
        lines: List[OCRLine],
        image_bytes: Optional[bytes] = None,
        image_mime: Optional[str] = None,
        use_image_context: bool = True,
    ) -> OCRMenuJudgeOutput:
        source_lines = self._normalize_lines(lines)
        if not source_lines:
            return OCRMenuJudgeOutput(items=[], menu_texts=[])

        if use_image_context and image_bytes and image_mime:
            selected_items = self._extract_menu_items_with_image(source_lines, image_bytes, image_mime)
        else:
            selected_items = self._extract_menu_items_text_only(source_lines)

        selected_keys = {self._norm(t) for t in selected_items}

        out_items: List[OCRTextLabel] = []
        for line in source_lines:
            key = self._norm(line.text)
            if key in selected_keys:
                item = OCRTextLabel(text=line.text, label="menu_item", is_menu=True)
            else:
                item = OCRTextLabel(text=line.text, label="other", is_menu=False)
            out_items.append(item)

        menu_texts = [it.text for it in out_items if it.label == "menu_item" and it.is_menu]
        return OCRMenuJudgeOutput(items=out_items, menu_texts=menu_texts)

    def _extract_menu_items_with_image(
        self,
        source_lines: List[OCRLine],
        image_bytes: bytes,
        image_mime: str,
    ) -> List[str]:
        payload = [line.text for line in source_lines]

        prompt = f"""
Extract only the food menu item names from this menu image.

Rules:
- Output only actual dish or drink names.
- Determine the primary language based on menu item names, not headers, descriptions, or prices.
- If the same menu item appears in multiple languages, output only one version in the primary language used for most menu item names on the menu.
- Exclude descriptions and explanatory text. These usually describe ingredients, flavor, cooking method, or serving style.
- Exclude prices, currency symbols, and standalone numeric lines.
- Exclude section headers, restaurant/store names, option/add-on text, and duplicate items.
- Preserve the exact text from the image. Do not translate, paraphrase, or invent names.

OCR text bundle (JSON array from OCR):
{payload}

Return JSON only:
{{
  "items": ["..."]
}}
""".strip()

        img_part = self.gemma.image_part_from_bytes(image_bytes, image_mime)
        raw = self.gemma.generate_text([prompt, img_part], max_output_tokens=1000)
        data = extract_first_json_object(raw)

        if data is None:
            retry_raw = self.gemma.generate_text(
                [f"{prompt}\nReturn ONLY one JSON object.", img_part],
                max_output_tokens=1000,
            )
            data = extract_first_json_object(retry_raw)

        return self._parse_items_array(data or {}, source_lines)

    def _extract_menu_items_text_only(self, source_lines: List[OCRLine]) -> List[str]:
        payload = [line.text for line in source_lines]
        prompt = f"""
Extract only the food menu item names from this OCR text bundle.
Exclude headers/descriptions/prices/options/duplicates and return only menu item names.

OCR text bundle (JSON array from OCR):
{payload}

Return JSON only:
{{
  "items": ["..."]
}}
""".strip()
        raw = self.gemma.generate_text([prompt], max_output_tokens=800)
        data = extract_first_json_object(raw)
        return self._parse_items_array(data or {}, source_lines)

    def _parse_items_array(self, data: dict, source_lines: List[OCRLine]) -> List[str]:
        raw_items = data.get("items", [])
        if not isinstance(raw_items, list):
            return []

        # OCR 원문 기준 exact 매핑을 우선 적용하고, 중복은 제거한다.
        source_map = {self._norm(ln.text): ln.text for ln in source_lines}
        out: List[str] = []
        seen = set()
        for item in raw_items:
            if not isinstance(item, str):
                continue
            key = self._norm(item)
            if not key or key in seen:
                continue
            if key in source_map:
                out.append(source_map[key])
                seen.add(key)
        return out

    @staticmethod
    def _normalize_lines(lines: List[OCRLine]) -> List[OCRLine]:
        out: List[OCRLine] = []
        seen = set()
        for line in lines:
            if not isinstance(line, OCRLine):
                continue
            text = " ".join((line.text or "").split()).strip()
            if not text:
                continue
            key = OCRMenuJudgeAgent._norm(text)
            if key in seen:
                continue
            seen.add(key)
            out.append(OCRLine(text=text, confidence=line.confidence, bbox=line.bbox))
        return out

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join((s or "").strip().split()).casefold()
