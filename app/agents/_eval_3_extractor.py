import re
from typing import List, Optional

from app.agents._0_contracts import OCRLine, OCRMenuJudgeOutput, OCRTextLabel
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import extract_first_json_object, normalize_list


def build_general_menu_classification_prompt(payload: list[dict], use_image_context: bool) -> str:
    image_rule = (
        "- Use the image only to resolve OCR ambiguity and very local layout relationships.\n"
        if use_image_context
        else "- Use OCR text only. Do not infer missing text.\n"
    )

    return f"""
Task:
Classify each OCR line from a photographed menu into exactly one label.

Goal:
Identify which lines are true final orderable menu items.

Core definition:
A menu_item is the main identity of a final sellable offering that a customer can directly choose from the menu.

A valid menu_item may be:
- a food item
- a drink
- a dessert or bakery item
- a combo or set
- a course menu
- a platter or shared item
- a named special or other directly orderable offering

A line is NOT a menu_item if it is primarily:
- a section or category heading
- a price
- a description
- an ingredient list
- a customization or modifier
- a variant selector
- a quantity, portion, or unit detail
- a promotion, recommendation marker, or badge
- store, service, or policy information
- other supporting text rather than the product identity itself

Allowed labels:
- menu_item
- category_header
- description
- price
- option_modifier
- quantity_portion
- promo
- store_info
- other

Decision process:
1. Does the line represent something a customer could directly choose and order?
2. Is it the main identity of the offering, rather than supporting information?
3. Would it still make sense as a product name if shown by itself?

Label as menu_item only if the answer is clearly yes.

Important rules:
- Preserve OCR text exactly in this step.
- Classify every line exactly once using the same index.
- Be conservative when uncertain.
- If a line mixes a product identity with secondary details, classify based on the primary role of the line.
- A line should be menu_item only when it is a clean orderable name. If description, slogan, taste notes, or explanatory copy are mixed in, prefer description/other.
- If a line mainly looks like a set/menu-board phrase rather than a concrete dish name, do NOT label it as menu_item.
- If a line contains a dish name plus a trailing standalone number or price fragment, treat the price part as secondary detail, not part of the menu identity.
- If a line is mostly a marketing phrase, set-title fragment, audience/serving-count phrase, or descriptive copy, do NOT label it as menu_item unless a concrete orderable product identity is clearly present.
- Prefer precision over recall: false positives are worse than missing a borderline menu item.
- Do not assume the menu belongs to a specific venue type (cafe/restaurant/bar/bakery/etc.).
- Do not rely on cuisine-specific assumptions.
- Work across restaurants, cafes, bakeries, bars, food courts, kiosks, and multilingual menus.
Visual grouping rule:
- OCR may split one visible menu name into multiple fragments.
- Only treat fragments as one menu name when they appear to be parts of the same inline product name on the same row or baseline.
- Do NOT merge vertically stacked title + description text, side badges, left/right gutter labels, section labels, or nearby explanatory copy into one menu item just because they are visually close.
- Do not invent unsupported text, but do use the image to understand whether fragmented OCR tokens are part of one orderable offering.
{image_rule}

OCR lines:
{payload}

Return JSON only:
{{
  "line_labels": [
    {{
      "index": 0,
      "text": "...",
      "label": "menu_item|category_header|description|price|option_modifier|quantity_portion|promo|store_info|other"
    }}
  ]
}}
""".strip()


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
        normalized_lines = self._normalize_lines(lines)
        if not normalized_lines:
            return OCRMenuJudgeOutput(items=[], menu_texts=[])
        source_lines = normalized_lines

        if use_image_context and image_bytes and image_mime:
            label_map = self._extract_labels_with_image(source_lines, image_bytes, image_mime)
        else:
            label_map = self._extract_labels_text_only(source_lines)

        out_items: List[OCRTextLabel] = []
        for line in source_lines:
            key = self._norm(line.text)
            label = label_map.get(key, "other")
            item = OCRTextLabel(text=line.text, label=label, is_menu=(label == "menu_item"))
            out_items.append(item)

        menu_texts = self._postprocess_menu_texts(
            [it.text for it in out_items if it.label == "menu_item" and it.is_menu]
        )
        return OCRMenuJudgeOutput(items=out_items, menu_texts=menu_texts)

    def _extract_labels_with_image(
        self,
        source_lines: List[OCRLine],
        image_bytes: bytes,
        image_mime: str,
    ) -> dict:
        payload = [{"index": idx, "text": line.text} for idx, line in enumerate(source_lines)]
        prompt = build_general_menu_classification_prompt(payload, use_image_context=True)

        img_part = self.gemma.image_part_from_bytes(image_bytes, image_mime)
        raw = self.gemma.generate_text([prompt, img_part], max_output_tokens=3000)
        data = extract_first_json_object(raw)

        if data is None:
            retry_raw = self.gemma.generate_text(
                [f"{prompt}\nReturn ONLY one JSON object.", img_part],
                max_output_tokens=3000,
            )
            data = extract_first_json_object(retry_raw)

        return self._parse_label_map(data or {}, source_lines)

    def _extract_labels_text_only(self, source_lines: List[OCRLine]) -> dict:
        payload = [{"index": idx, "text": line.text} for idx, line in enumerate(source_lines)]
        prompt = build_general_menu_classification_prompt(payload, use_image_context=False)
        raw = self.gemma.generate_text([prompt], max_output_tokens=3000)
        data = extract_first_json_object(raw)
        if data is None:
            retry_raw = self.gemma.generate_text(
                [f"{prompt}\nReturn ONLY one JSON object."],
                max_output_tokens=3000,
            )
            data = extract_first_json_object(retry_raw)
        return self._parse_label_map(data or {}, source_lines)

    def _parse_label_map(self, data: dict, source_lines: List[OCRLine]) -> dict:
        allowed_labels = {
            "menu_item",
            "category_header",
            "description",
            "price",
            "option_modifier",
            "quantity_portion",
            "promo",
            "store_info",
            "other",
        }
        legacy_map = {
            "option": "option_modifier",
            "temperature": "option_modifier",
            "size_volume": "quantity_portion",
        }
        source_keys = {self._norm(ln.text) for ln in source_lines}
        out = {}

        raw_lines = data.get("line_labels", [])
        if isinstance(raw_lines, list):
            for item in raw_lines:
                if not isinstance(item, dict):
                    continue
                idx = self._to_index(item.get("index"))
                text = item.get("text", "")
                label = str(item.get("label", "other")).strip().casefold()
                label = legacy_map.get(label, label)
                if label not in allowed_labels:
                    label = "other"

                if idx is not None and 0 <= idx < len(source_lines):
                    key = self._norm(source_lines[idx].text)
                    out[key] = label
                    continue

                # index 누락 시에만 텍스트 매핑 fallback
                key = self._norm(text)
                if key in source_keys:
                    out[key] = label

        if out:
            return out

        # 이전 포맷(items 배열)도 계속 호환한다.
        selected_items = self._parse_items_array(data, source_lines)
        return {self._norm(t): "menu_item" for t in selected_items}

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
    def _postprocess_menu_texts(menu_texts: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()

        for text in menu_texts:
            if not isinstance(text, str):
                continue
            cleaned = " ".join(text.split()).strip()
            if not cleaned:
                continue
            cleaned = OCRMenuJudgeAgent._strip_trailing_price(cleaned)
            if not cleaned:
                continue
            key = OCRMenuJudgeAgent._norm(cleaned)
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)

        return out

    @staticmethod
    def _strip_trailing_price(text: str) -> str:
        return re.sub(r"\s+\d+(?:\.\d+)?$", "", text).strip()

    @staticmethod
    def _norm(s: str) -> str:
        return " ".join((s or "").strip().split()).casefold()

    @staticmethod
    def _to_index(v) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None
