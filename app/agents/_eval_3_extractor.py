from typing import List, Optional

from app.agents._0_contracts import OCRLine, OCRMenuJudgeOutput, OCRTextLabel
from app.clients.gemma_client import GemmaClient
from app.utils.parsing import extract_first_json_object, normalize_list


def build_general_menu_classification_prompt(payload: list[dict], use_image_context: bool) -> str:
    image_rule = (
        "- Use the image only to resolve OCR ambiguity, visual grouping, and layout relationships.\n"
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
- A line can still be menu_item even if it includes limited attached detail, as long as it still clearly expresses one complete orderable offering.
- Do not assume the menu belongs to a specific venue type (cafe/restaurant/bar/bakery/etc.).
- Do not rely on cuisine-specific assumptions.
- Work across restaurants, cafes, bakeries, bars, food courts, kiosks, and multilingual menus.
Visual grouping rule:
- OCR may split one visible menu name into multiple fragments.
- When the image clearly shows that adjacent fragments belong to one menu name on the same row or visual group, use that visual evidence when deciding whether the fragments represent a menu item.
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
        source_lines = self._reconstruct_lines_with_bbox(normalized_lines)
        if not source_lines:
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

        menu_texts = [it.text for it in out_items if it.label == "menu_item" and it.is_menu]
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

    def _reconstruct_lines_with_bbox(self, lines: List[OCRLine]) -> List[OCRLine]:
        # bbox는 구조 복원(열/행/토큰 병합)에만 사용하고 LLM payload에는 포함하지 않는다.
        boxed = [ln for ln in lines if self._has_bbox(ln)]
        if len(boxed) < 2:
            return lines

        columns = self._group_by_column(boxed)
        merged: List[OCRLine] = []
        for col in columns:
            rows = self._group_by_row(col)
            for row in rows:
                row_sorted = sorted(row, key=lambda ln: self._left_x(ln.bbox))
                merged_text = self._merge_row_tokens(row_sorted)
                if not merged_text:
                    continue
                conf = sum(max(0.0, min(1.0, float(ln.confidence))) for ln in row_sorted) / max(1, len(row_sorted))
                merged.append(OCRLine(text=merged_text, confidence=conf, bbox=[]))

        if not merged:
            return lines
        return self._normalize_lines(merged)

    def _group_by_column(self, lines: List[OCRLine]) -> List[List[OCRLine]]:
        # 좌우로 크게 벌어진 경우를 다른 column으로 본다.
        enriched = []
        widths = []
        for ln in lines:
            left = self._left_x(ln.bbox)
            right = self._right_x(ln.bbox)
            cx = (left + right) / 2.0
            widths.append(max(1.0, right - left))
            enriched.append((cx, ln))
        if not enriched:
            return []

        median_w = sorted(widths)[len(widths) // 2]
        split_gap = max(50.0, median_w * 1.8)
        enriched.sort(key=lambda x: x[0])

        cols: List[List[OCRLine]] = []
        cur: List[OCRLine] = []
        prev_cx: Optional[float] = None
        for cx, ln in enriched:
            if prev_cx is None or abs(cx - prev_cx) <= split_gap:
                cur.append(ln)
            else:
                if cur:
                    cols.append(cur)
                cur = [ln]
            prev_cx = cx
        if cur:
            cols.append(cur)

        cols.sort(key=lambda c: min((self._left_x(ln.bbox) for ln in c), default=0.0))
        return cols

    def _group_by_row(self, lines: List[OCRLine]) -> List[List[OCRLine]]:
        enriched = []
        heights = []
        for ln in lines:
            top = self._top_y(ln.bbox)
            bottom = self._bottom_y(ln.bbox)
            cy = (top + bottom) / 2.0
            h = max(8.0, bottom - top)
            heights.append(h)
            enriched.append((cy, ln))
        if not enriched:
            return []

        median_h = sorted(heights)[len(heights) // 2]
        row_gap = max(10.0, median_h * 0.7)
        enriched.sort(key=lambda x: x[0])

        rows: List[List[OCRLine]] = []
        row_centers: List[float] = []
        for cy, ln in enriched:
            if not rows:
                rows.append([ln])
                row_centers.append(cy)
                continue

            if abs(cy - row_centers[-1]) <= row_gap:
                rows[-1].append(ln)
                n = float(len(rows[-1]))
                row_centers[-1] = ((row_centers[-1] * (n - 1.0)) + cy) / n
            else:
                rows.append([ln])
                row_centers.append(cy)
        return rows

    def _merge_row_tokens(self, row_sorted: List[OCRLine]) -> str:
        tokens = [self._clean_token(ln.text) for ln in row_sorted if self._clean_token(ln.text)]
        if not tokens:
            return ""
        if len(tokens) == 1:
            return tokens[0]

        out = [tokens[0]]
        for i in range(1, len(tokens)):
            prev_ln = row_sorted[i - 1]
            cur_ln = row_sorted[i]
            cur_tok = tokens[i]

            gap = self._left_x(cur_ln.bbox) - self._right_x(prev_ln.bbox)
            gap = max(0.0, gap)

            prev_tok = self._clean_token(prev_ln.text)
            join_no_space = (
                (self._is_single_hangul_char(prev_tok) and self._is_single_hangul_char(cur_tok) and gap <= 16.0)
                or gap <= 2.0
            )
            if join_no_space:
                out[-1] = out[-1] + cur_tok
            else:
                out.append(cur_tok)
        return " ".join(out).strip()

    @staticmethod
    def _clean_token(text: str) -> str:
        return " ".join((text or "").split()).strip()

    @staticmethod
    def _is_single_hangul_char(token: str) -> bool:
        if len(token) != 1:
            return False
        ch = token[0]
        return "\uac00" <= ch <= "\ud7a3"

    @staticmethod
    def _has_bbox(line: OCRLine) -> bool:
        return bool(line.bbox and len(line.bbox) >= 2)

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

    @staticmethod
    def _to_index(v) -> Optional[int]:
        try:
            if v is None:
                return None
            return int(v)
        except Exception:
            return None

    @staticmethod
    def _top_y(bbox: List[List[float]]) -> float:
        return min((p[1] for p in bbox), default=0.0)

    @staticmethod
    def _bottom_y(bbox: List[List[float]]) -> float:
        return max((p[1] for p in bbox), default=0.0)

    @staticmethod
    def _left_x(bbox: List[List[float]]) -> float:
        return min((p[0] for p in bbox), default=0.0)

    @staticmethod
    def _right_x(bbox: List[List[float]]) -> float:
        return max((p[0] for p in bbox), default=0.0)
