import re
from statistics import median
from typing import List, Optional

from app.agents._0_contracts import OCRLine, OCRMenuJudgeOutput, OCRTextLabel
from app.clients.gemma_client import GemmaClient
from app.utils.menu_item_cleaner import clean_menu_candidates
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
Visual emphasis rule:
- Within the same local menu block, a larger, bolder, darker, or more visually prominent line is more likely to be the true menu_item.
- Within the same local menu block, a smaller, lighter, denser, or lower-contrast line is more likely to be description, option_modifier, quantity_portion, or other supporting text.
- If a prominent title line is followed by one or more smaller or lighter lines, prefer labeling only the prominent title as menu_item and the following lines as description or option text unless they are clearly separate products.
- Use visual emphasis only as a relative local cue. Do not assume every large line is a menu item or every small line is a description without supporting context.
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

    SPANISH_LANGS = {"es"}
    KOREAN_LANGS = {"ko", "korean", "kr"}
    SPANISH_PRICE_SUFFIX_RE = re.compile(
        r"(?:(?:[|,:]\s*|\s+)\d{1,3}(?:[.,]\d+)?|(?<=\D)\d{1,3}(?:[.,]\d+)?)\s*(?:[A-Z]{1,8}(?:/[A-Z]{1,8})*)?\s*$",
        re.IGNORECASE,
    )
    SPANISH_PRICE_TOKEN_RE = re.compile(
        r"(?:[|,:]\s*|\s+|(?<=\D))\d{1,3}(?:[.,]\d+)?(?:\s*[A-Z]{1,8}(?:/[A-Z]{1,8})*)?",
        re.IGNORECASE,
    )
    SPANISH_STANDALONE_PRICE_RE = re.compile(
        r"^[|,:]?\s*\d{1,3}(?:[.,]\d+)?\s*(?:[A-Z]{1,8}(?:/[A-Z]{1,8})*)?\s*$",
        re.IGNORECASE,
    )
    SPANISH_DESCRIPTION_CUES = (
        "sauteed",
        "served",
        "stuffed",
        "traditional",
        "homemade",
        "delicious",
        "touch of",
        "topped with",
        "mixed with",
        "deep fried",
        "fried",
        "olive oil",
        "white wine",
        "sauce",
        "garlic",
    )

    CHINESE_PRICE_RE = re.compile(r"^(?:[$¥]?\s*)?(?:\d+(?:[.,]\d+)?\s*元?|元+|\d+)$")
    CHINESE_INDEX_RE = re.compile(r"^[\(\[（【]?\s*\d{1,3}\s*[\)\]）】]?$")
    CHINESE_STORE_RE = re.compile(r"(欢迎|歡迎|www\.|nipic|昵图网|by[:：]|no[:：])", re.IGNORECASE)
    CHINESE_HEADER_RE = re.compile(r"^(?:菜谱|菜單|菜单|.+类)$")
    CHINESE_HEADER_WORDS = {
        "菜谱",
        "菜單",
        "菜单",
        "热菜",
        "凉菜",
        "燒烤",
        "烧烤",
        "汤类",
        "湯類",
        "汤",
        "湯",
        "肉类",
        "肉類",
        "素类",
        "素類",
        "海鲜类",
        "海鮮類",
        "主食",
        "酒水",
        "饮品",
        "飲品",
    }
    CHINESE_MENU_METHOD_CUES = {
        "炒",
        "烧",
        "燒",
        "烤",
        "炖",
        "燉",
        "煸",
        "炸",
        "拌",
        "煮",
        "煲",
        "锅",
        "鍋",
        "汤",
        "湯",
        "麻辣",
        "红烧",
        "紅燒",
        "家常",
        "回锅",
        "回鍋",
        "明炉",
        "明爐",
        "锡纸",
        "錫紙",
        "凉拌",
        "涼拌",
    }
    CHINESE_MENU_FOOD_CUES = {
        "羊",
        "牛",
        "鸡",
        "雞",
        "鸭",
        "鴨",
        "鱼",
        "魚",
        "虾",
        "蝦",
        "肉",
        "肠",
        "腸",
        "腰",
        "肚",
        "肝",
        "翅",
        "腿",
        "脖",
        "头",
        "頭",
        "珍",
        "宝",
        "寶",
        "鞭",
        "串",
        "筋",
        "骨",
        "尾",
        "排骨",
        "龙虾",
        "龍蝦",
        "豆腐",
        "豆角",
        "土豆",
        "木耳",
        "玉米",
        "茄子",
        "蘑菇",
        "香菇",
        "香蘑",
        "青瓜",
        "黄瓜",
        "黃瓜",
        "油麦菜",
        "油麥菜",
        "生菜",
        "鸡蛋",
        "雞蛋",
        "疙瘩",
        "菌",
        "面",
        "麵",
    }

    def __init__(self, gemma: GemmaClient):
        self.gemma = gemma

    def run(self, texts: List[str], ocr_lang: Optional[str] = None) -> OCRMenuJudgeOutput:
        candidates = normalize_list(texts, limit=300)
        lines = [OCRLine(text=t, confidence=1.0, bbox=[]) for t in candidates]
        return self.run_lines(lines, ocr_lang=ocr_lang)

    def run_lines(self, lines: List[OCRLine], ocr_lang: Optional[str] = None) -> OCRMenuJudgeOutput:
        return self.run_lines_with_image(lines=lines, image_bytes=None, image_mime=None, ocr_lang=ocr_lang)

    def run_lines_with_image(
        self,
        lines: List[OCRLine],
        image_bytes: Optional[bytes] = None,
        image_mime: Optional[str] = None,
        use_image_context: bool = True,
        ocr_lang: Optional[str] = None,
    ) -> OCRMenuJudgeOutput:
        normalized_lines = self._normalize_lines(lines)
        if not normalized_lines:
            return OCRMenuJudgeOutput(items=[], menu_texts=[])
        source_lines = normalized_lines
        normalized_lang = (ocr_lang or "").strip().lower()

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

        menu_candidates = [it.text for it in out_items if it.label == "menu_item" and it.is_menu]
        if normalized_lang in self.SPANISH_LANGS:
            # Prefer deterministic price-bearing title extraction for Spanish menus.
            # This avoids description bleed from LLM labels on bilingual ES/EN boards.
            spanish_rule_items = self._collect_spanish_title_candidates(source_lines)
            if spanish_rule_items:
                menu_candidates = spanish_rule_items
        elif normalized_lang in self.KOREAN_LANGS:
            # Google Vision OCR는 한국어에서 "메뉴명 + 설명 꼬리"가 한 줄로 붙는 경우가 많다.
            # LLM이 description으로 분류한 줄에서 메뉴명 prefix를 보수적으로 복구한다.
            korean_rule_items = self._collect_korean_title_candidates(source_lines, out_items)
            if korean_rule_items:
                menu_candidates = self._merge_menu_candidates(menu_candidates, korean_rule_items)

        menu_texts = self._postprocess_menu_texts(menu_candidates)
        return OCRMenuJudgeOutput(items=out_items, menu_texts=menu_texts)

    @classmethod
    def _collect_korean_title_candidates(
        cls,
        source_lines: List[OCRLine],
        labeled_items: List[OCRTextLabel],
    ) -> List[str]:
        out: List[str] = []
        seen = set()

        for line, labeled in zip(source_lines, labeled_items):
            if labeled.label == "menu_item":
                continue

            clean_result = clean_menu_candidates([line.text])
            if not clean_result.cleaned_items:
                continue

            candidate = clean_result.cleaned_items[0]
            if cls._norm(candidate) == cls._norm(line.text):
                # 분류기와 동일하게 보이는 줄은 복구 근거가 약해 제외한다.
                continue

            key = cls._norm(candidate)
            if key in seen:
                continue
            seen.add(key)
            out.append(candidate)

        return out

    @classmethod
    def _merge_menu_candidates(cls, base: List[str], recovered: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()

        for text in [*base, *recovered]:
            key = cls._norm(text)
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(text)
        return out

    @classmethod
    def _collect_spanish_title_candidates(cls, source_lines: List[OCRLine]) -> List[str]:
        out: List[str] = []
        seen = set()
        for idx, line in enumerate(source_lines):
            title, has_price, may_continue = cls._extract_spanish_title_from_line(line.text)
            if has_price:
                if not cls._is_spanish_menu_title_candidate(title):
                    continue
            else:
                # Recover menu names when OCR splits title and price into separate lines
                # (e.g., "Empanadillas" + next line "13").
                if not cls._is_spanish_menu_title_candidate(title):
                    continue
                if not cls._has_nearby_spanish_standalone_price(source_lines, idx):
                    continue

            merged = title
            if has_price and may_continue:
                continuation = cls._find_spanish_title_continuation(source_lines, idx)
                if continuation:
                    merged = f"{merged} {continuation}".strip()

            key = cls._norm(merged)
            if key in seen:
                continue
            seen.add(key)
            out.append(merged)
        return out

    @classmethod
    def _normalize_spanish_title_candidate(cls, text: str) -> str:
        title, _, _ = cls._extract_spanish_title_from_line(text)
        return title

    @classmethod
    def _extract_spanish_title_from_line(cls, text: str) -> tuple[str, bool, bool]:
        normalized = " ".join((text or "").split()).strip()
        if not normalized:
            return "", False, False

        inline_match = cls.SPANISH_PRICE_TOKEN_RE.search(normalized)
        if inline_match is not None:
            raw_title = normalized[: inline_match.start()].strip()
            if raw_title:
                trailing = normalized[inline_match.end() :].strip()
                marker_char = normalized[inline_match.start()] if inline_match.start() < len(normalized) else ""
                may_continue = raw_title.endswith((",", "/", "&")) or marker_char in {",", "/", "&"}
                normalized = raw_title.rstrip(".,:;|/\\- ").strip()
                inline_continuation = cls._normalize_spanish_inline_continuation(trailing)
                if inline_continuation:
                    normalized = f"{normalized} {inline_continuation}".strip()
                    may_continue = False
                return normalized, True, may_continue

        match = cls.SPANISH_PRICE_SUFFIX_RE.search(normalized)
        if match is not None:
            raw_title = normalized[: match.start()].strip()
            marker_char = normalized[match.start()] if match.start() < len(normalized) else ""
            may_continue = raw_title.endswith((",", "/", "&")) or marker_char in {",", "/", "&"}
            normalized = raw_title.rstrip(".,:;|/\\- ").strip()
            return normalized, True, may_continue

        normalized = normalized.rstrip(".,:;|/\\- ").strip()
        return normalized, False, False

    @classmethod
    def _find_spanish_title_continuation(cls, source_lines: List[OCRLine], start_idx: int) -> str:
        # OCR ordering can interleave columns, so we check a small lookahead window.
        for offset in (1, 2, 3):
            idx = start_idx + offset
            if idx >= len(source_lines):
                break
            continuation = cls._normalize_spanish_title_continuation(source_lines[idx].text)
            if continuation:
                return continuation
        return ""

    @classmethod
    def _is_spanish_menu_title_candidate(cls, text: str) -> bool:
        if not text:
            return False
        if len(text) < 3 or len(text) > 52:
            return False
        if text.endswith("."):
            return False

        lowered = text.casefold()
        if any(cue in lowered for cue in cls.SPANISH_DESCRIPTION_CUES):
            return False

        words = text.split()
        if len(words) > 8:
            return False
        if not any(ch.isalpha() for ch in text):
            return False
        letters = [ch for ch in text if ch.isalpha()]
        if letters and len(text) > 12:
            uppercase_ratio = sum(1 for ch in letters if ch.isupper()) / float(len(letters))
            if uppercase_ratio >= 0.9:
                return False

        first = words[0] if words else ""
        if first and first[0].isalpha() and first[0].islower() and first.casefold() not in {"de", "y", "con"}:
            return False
        return True

    @classmethod
    def _normalize_spanish_title_continuation(cls, text: str) -> str:
        continuation = " ".join((text or "").split()).strip()
        if not continuation:
            return ""
        _, has_price, _ = cls._extract_spanish_title_from_line(continuation)
        if has_price:
            return ""
        if len(continuation) > 28:
            return ""
        lowered = continuation.casefold()
        if any(cue in lowered for cue in cls.SPANISH_DESCRIPTION_CUES):
            return ""
        if continuation.endswith("."):
            return ""
        if not any(ch.isalpha() for ch in continuation):
            return ""

        words = continuation.split()
        if len(words) > 3:
            return ""
        if continuation.startswith("("):
            return ""
        if not any(token in continuation for token in (",", " y ", "&", "/")):
            return ""
        if continuation[0].isalpha() and not continuation[0].isupper():
            return ""

        continuation = continuation.rstrip(".,:;|/\\- ").strip()
        return continuation

    @classmethod
    def _normalize_spanish_inline_continuation(cls, text: str) -> str:
        continuation = " ".join((text or "").split()).strip()
        if not continuation:
            return ""
        if len(continuation) > 32:
            return ""
        lowered = continuation.casefold()
        if any(cue in lowered for cue in cls.SPANISH_DESCRIPTION_CUES):
            return ""
        if continuation.endswith("."):
            return ""
        if continuation.startswith("("):
            return ""
        if not any(ch.isalpha() for ch in continuation):
            return ""

        words = continuation.split()
        if len(words) > 4:
            return ""
        first = words[0].casefold() if words else ""
        if first not in {"de", "y", "con"} and words and words[0][0].isalpha() and words[0][0].islower():
            return ""

        continuation = continuation.lstrip(",/& ").strip()
        continuation = continuation.rstrip(".,:;|/\\- ").strip()
        return continuation

    @classmethod
    def _has_nearby_spanish_standalone_price(cls, source_lines: List[OCRLine], center_idx: int) -> bool:
        for offset in (-1, 1, -2, 2, 3):
            idx = center_idx + offset
            if idx < 0 or idx >= len(source_lines):
                continue
            candidate = " ".join((source_lines[idx].text or "").split()).strip()
            if cls._is_spanish_standalone_price_line(candidate):
                return True
        return False

    @classmethod
    def _is_spanish_standalone_price_line(cls, text: str) -> bool:
        if not text:
            return False
        return bool(cls.SPANISH_STANDALONE_PRICE_RE.fullmatch(text))

    @classmethod
    def _run_chinese_layout_strategy(cls, source_lines: List[OCRLine]) -> OCRMenuJudgeOutput:
        items: List[OCRTextLabel] = []
        for line in source_lines:
            label = cls._classify_chinese_line(line)
            items.append(OCRTextLabel(text=line.text, label=label, is_menu=(label == "menu_item")))

        candidates = cls._collect_chinese_direct_candidates(source_lines)
        candidates.extend(cls._recover_chinese_vertical_candidates(source_lines))
        candidates.sort(key=lambda item: (item[1], item[2], cls._norm(item[0])))

        menu_texts = cls._postprocess_chinese_menu_texts([text for text, _, _ in candidates])
        return OCRMenuJudgeOutput(items=items, menu_texts=menu_texts)

    @classmethod
    def _classify_chinese_line(cls, line: OCRLine) -> str:
        text = cls._compact_text(line.text)
        if not text:
            return "other"
        if cls._is_chinese_store_info(text):
            return "store_info"
        if cls._is_chinese_header(text):
            return "category_header"
        if cls._is_chinese_price(text):
            return "price"
        if cls._is_chinese_index(text):
            return "quantity_portion"
        if cls._is_horizontal_chinese_menu_candidate(line):
            return "menu_item"
        return "other"

    @classmethod
    def _collect_chinese_direct_candidates(cls, source_lines: List[OCRLine]) -> List[tuple[str, float, float]]:
        candidates: List[tuple[str, float, float]] = []
        for line in source_lines:
            if not cls._is_horizontal_chinese_menu_candidate(line):
                continue
            left, top, _, _, _, _ = cls._bbox_rect(line.bbox)
            cleaned = cls._clean_chinese_candidate_text(line.text)
            if not cls._is_valid_chinese_menu_candidate(cleaned):
                continue
            candidates.append((cleaned, top, left))
        return candidates

    @classmethod
    def _recover_chinese_vertical_candidates(cls, source_lines: List[OCRLine]) -> List[tuple[str, float, float]]:
        char_tokens, char_heights, char_widths = cls._build_chinese_char_tokens(source_lines)
        if len(char_tokens) < 4:
            return []

        row_tol = max(10.0, (median(char_heights) if char_heights else 16.0) * 0.75)
        gap_threshold = max(34.0, (median(char_widths) if char_widths else 18.0) * 1.8)
        section_ranges = cls._build_chinese_section_ranges(source_lines)

        candidates: List[tuple[str, float, float]] = []
        for left_bound, right_bound in section_ranges:
            section_tokens = [
                token for token in char_tokens
                if left_bound <= token["x"] <= right_bound
            ]
            if len(section_tokens) < 2:
                continue
            section_tokens.sort(key=lambda token: token["y"])

            rows: List[dict] = []
            for token in section_tokens:
                if not rows or abs(token["y"] - rows[-1]["y"]) > row_tol:
                    rows.append({"y": token["y"], "tokens": [token]})
                else:
                    rows[-1]["tokens"].append(token)
                    count = len(rows[-1]["tokens"])
                    rows[-1]["y"] = ((rows[-1]["y"] * (count - 1)) + token["y"]) / count

            for row in rows:
                tokens = sorted(row["tokens"], key=lambda token: token["x"])
                segments = cls._split_chinese_row_tokens(tokens, gap_threshold)
                for segment in segments:
                    if not segment:
                        continue
                    text = "".join(token["ch"] for token in segment)
                    cleaned = cls._clean_chinese_candidate_text(text)
                    if not cls._is_valid_chinese_menu_candidate(cleaned):
                        continue
                    candidates.append((cleaned, row["y"], segment[0]["x"]))
        return candidates

    @classmethod
    def _build_chinese_char_tokens(
        cls,
        source_lines: List[OCRLine],
    ) -> tuple[List[dict], List[float], List[float]]:
        tokens: List[dict] = []
        char_heights: List[float] = []
        char_widths: List[float] = []

        for line in source_lines:
            text = cls._compact_text(line.text)
            if not text or cls._is_chinese_store_info(text) or cls._is_chinese_header(text) or cls._is_chinese_price(text):
                continue
            if cls._is_horizontal_chinese_menu_candidate(line):
                continue

            left, top, right, bottom, width, height = cls._bbox_rect(line.bbox)
            han_chars = [ch for ch in text if cls._is_chinese_candidate_char(ch)]
            if not han_chars:
                continue

            if cls._is_vertical_chinese_text_line(line):
                step = max(1.0, height / max(len(han_chars), 1))
                x_center = (left + right) / 2.0
                for idx, ch in enumerate(han_chars):
                    tokens.append(
                        {
                            "x": x_center,
                            "y": top + ((idx + 0.5) * step),
                            "ch": ch,
                            "w": width,
                            "h": step,
                        }
                    )
                    char_heights.append(step)
                    char_widths.append(width)
                continue

            if len(han_chars) <= 3:
                step = max(1.0, width / max(len(han_chars), 1))
                y_center = (top + bottom) / 2.0
                for idx, ch in enumerate(han_chars):
                    tokens.append(
                        {
                            "x": left + ((idx + 0.5) * step),
                            "y": y_center,
                            "ch": ch,
                            "w": step,
                            "h": height,
                        }
                    )
                    char_heights.append(height)
                    char_widths.append(step)

        return tokens, char_heights, char_widths

    @classmethod
    def _build_chinese_section_ranges(cls, source_lines: List[OCRLine]) -> List[tuple[float, float]]:
        header_centers: List[float] = []
        for line in source_lines:
            text = cls._compact_text(line.text)
            if not cls._is_chinese_header(text):
                continue
            if not text.endswith("类"):
                continue
            left, _, right, _, _, _ = cls._bbox_rect(line.bbox)
            header_centers.append((left + right) / 2.0)

        if not header_centers:
            return [(-10**9, 10**9)]

        header_centers.sort()
        merged: List[float] = []
        for center in header_centers:
            if not merged or abs(center - merged[-1]) > 70.0:
                merged.append(center)
            else:
                merged[-1] = (merged[-1] + center) / 2.0

        if len(merged) == 1:
            return [(-10**9, 10**9)]

        boundaries: List[float] = [-10**9]
        for idx in range(len(merged) - 1):
            boundaries.append((merged[idx] + merged[idx + 1]) / 2.0)
        boundaries.append(10**9)
        return list(zip(boundaries[:-1], boundaries[1:]))

    @staticmethod
    def _split_chinese_row_tokens(tokens: List[dict], gap_threshold: float) -> List[List[dict]]:
        if not tokens:
            return []
        segments: List[List[dict]] = [[tokens[0]]]
        for prev, cur in zip(tokens, tokens[1:]):
            if (cur["x"] - prev["x"]) > gap_threshold:
                segments.append([cur])
                continue
            segments[-1].append(cur)
        return segments

    @classmethod
    def _postprocess_chinese_menu_texts(cls, menu_texts: List[str]) -> List[str]:
        out: List[str] = []
        seen = set()
        for text in menu_texts:
            cleaned = cls._clean_chinese_candidate_text(text)
            if not cls._is_valid_chinese_menu_candidate(cleaned):
                continue
            key = cls._norm(cleaned)
            if key in seen:
                continue
            seen.add(key)
            out.append(cleaned)
        return out

    @classmethod
    def _is_horizontal_chinese_menu_candidate(cls, line: OCRLine) -> bool:
        text = cls._compact_text(line.text)
        if not text:
            return False
        if cls._is_chinese_store_info(text) or cls._is_chinese_header(text) or cls._is_chinese_price(text) or cls._is_chinese_index(text):
            return False
        if not cls._has_han(text):
            return False

        _, _, _, _, width, height = cls._bbox_rect(line.bbox)
        han_count = cls._han_char_count(text)
        if han_count < 2 or han_count > 8:
            return False
        if height > 0 and width < (height * 1.1):
            return False
        return cls._is_valid_chinese_menu_candidate(text)

    @classmethod
    def _is_vertical_chinese_text_line(cls, line: OCRLine) -> bool:
        text = cls._compact_text(line.text)
        if not text or not cls._has_han(text):
            return False
        if cls._is_chinese_store_info(text) or cls._is_chinese_header(text) or cls._is_chinese_price(text):
            return False
        _, _, _, _, width, height = cls._bbox_rect(line.bbox)
        return height > (width * 1.3) and len(text) >= 2

    @classmethod
    def _clean_chinese_candidate_text(cls, text: str) -> str:
        compact = cls._compact_text(text)
        if not compact:
            return ""
        compact = re.sub(r"[（）()\[\]【】·.,，:：;；、/\\]+", "", compact)
        compact = compact.replace("元", "")
        compact = re.sub(r"\d+", "", compact)
        return compact.strip()

    @classmethod
    def _is_valid_chinese_menu_candidate(cls, text: str) -> bool:
        compact = cls._clean_chinese_candidate_text(text)
        if not compact:
            return False
        if cls._is_chinese_store_info(compact) or cls._is_chinese_header(compact):
            return False
        han_count = cls._han_char_count(compact)
        if han_count < 2 or han_count > 8:
            return False
        if cls._repeated_char_ratio(compact) > 0.6:
            return False
        if not cls._has_chinese_menu_cue(compact):
            return False
        return True

    @classmethod
    def _is_chinese_store_info(cls, text: str) -> bool:
        compact = cls._compact_text(text)
        if not compact:
            return False
        if cls.CHINESE_STORE_RE.search(compact):
            return True
        return "欢迎您" in compact or "歡迎您" in compact

    @classmethod
    def _is_chinese_header(cls, text: str) -> bool:
        compact = cls._compact_text(text)
        if not compact:
            return False
        if compact in cls.CHINESE_HEADER_WORDS:
            return True
        return cls.CHINESE_HEADER_RE.fullmatch(compact) is not None

    @classmethod
    def _is_chinese_price(cls, text: str) -> bool:
        compact = cls._compact_text(text)
        if not compact:
            return False
        return cls.CHINESE_PRICE_RE.fullmatch(compact) is not None

    @classmethod
    def _is_chinese_index(cls, text: str) -> bool:
        compact = cls._compact_text(text)
        if not compact:
            return False
        return cls.CHINESE_INDEX_RE.fullmatch(compact) is not None

    @staticmethod
    def _compact_text(text: str) -> str:
        return "".join((text or "").split()).strip()

    @staticmethod
    def _has_han(text: str) -> bool:
        return any(0x4E00 <= ord(ch) <= 0x9FFF for ch in text or "")

    @staticmethod
    def _han_char_count(text: str) -> int:
        return sum(1 for ch in text or "" if 0x4E00 <= ord(ch) <= 0x9FFF)

    @staticmethod
    def _is_chinese_candidate_char(ch: str) -> bool:
        return (0x4E00 <= ord(ch) <= 0x9FFF) and ch not in {"元", "类"}

    @staticmethod
    def _repeated_char_ratio(text: str) -> float:
        if not text:
            return 0.0
        counts = {}
        for ch in text:
            counts[ch] = counts.get(ch, 0) + 1
        return max(counts.values(), default=0) / float(len(text))

    @classmethod
    def _has_chinese_menu_cue(cls, text: str) -> bool:
        compact = cls._clean_chinese_candidate_text(text)
        if not compact:
            return False
        if any(token in compact for token in cls.CHINESE_MENU_METHOD_CUES):
            return True
        if any(token in compact for token in cls.CHINESE_MENU_FOOD_CUES):
            return True
        return False

    @staticmethod
    def _bbox_rect(bbox: List[List[float]]) -> tuple[float, float, float, float, float, float]:
        if not bbox:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        xs = [point[0] for point in bbox]
        ys = [point[1] for point in bbox]
        left = min(xs)
        right = max(xs)
        top = min(ys)
        bottom = max(ys)
        return left, top, right, bottom, max(0.0, right - left), max(0.0, bottom - top)

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
