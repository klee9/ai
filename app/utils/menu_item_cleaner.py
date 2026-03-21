import re
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from typing import Dict, List


LEADING_INDEX_RE = re.compile(
    r"^\s*(?:[\(\[]?\d{1,3}[\)\].:,-]?\s+|[A-Za-z][\).:-]\s+|[ivxlcdm]+\.\s+)",
    re.IGNORECASE,
)
TRAILING_PRICE_RE_LIST = [
    re.compile(r"\s+[$€£¥₩]\s*\d+(?:[.,]\d{1,2})?$"),
    re.compile(
        r"\s+(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d{1,2})?\s*(?:원|krw|usd|eur|cny|jpy|元|円)$",
        re.IGNORECASE,
    ),
    re.compile(r"\s+(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d{1,2})$"),
    re.compile(r"(?<=\D)(?:\d{1,3}(?:,\d{3})+|\d+)(?:[.,]\d{1,2})$"),
]
EDGE_PUNCT_RE = re.compile(r"^[\s\-–—:;,.|/\\•·*]+|[\s\-–—:;,.|/\\•·*]+$")
NON_TEXT_RE = re.compile(r"[\s\-_–—:;,.|/\\•·*()\[\]{}<>]+")
VOLUME_SUFFIX_RE = re.compile(r"(?<=\d)m\b", re.IGNORECASE)
VOLUME_LIKE_RE = re.compile(r"(?:\d+(?:\.\d+)?)\s*(?:ml|l|oz|g|kg|mg|인분|잔|병|캔|pc|pcs)\b", re.IGNORECASE)
REPEATED_BRAND_RE = re.compile(r"^([a-z가-힣])\1+$", re.IGNORECASE)
BADGE_TOKEN_STRIP_RE = re.compile(r"^[\s\[\]\(\)\{\}<>【】「」'\"`~!@#$%^&*_+=|\\:;,.?/·★☆\-]+|[\s\[\]\(\)\{\}<>【】「」'\"`~!@#$%^&*_+=|\\:;,.?/·★☆\-]+$")
SINGLE_COMMA_JOINER_SUFFIX_RE = re.compile(
    r",\s*(?:y|e|and|&)\s+[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’/\-]*(?:\s+[A-Za-zÀ-ÖØ-öø-ÿ][A-Za-zÀ-ÖØ-öø-ÿ'’/\-]*)?$",
    re.IGNORECASE,
)
MENU_BADGE_TOKENS = {
    "best",
    "hot",
    "new",
    "베스트",
    "추천",
}
ALCOHOL_STRENGTH_RE = re.compile(r"(?:\(\s*)?\d+(?:\.\d+)?\s*(?:%|℃|도)(?:\s*\))?", re.IGNORECASE)
KOREAN_TITLE_DESCRIPTION_SPLIT_RE = re.compile(
    r"^(.{2,32}?(?:솥밥|전골|스키야키|볶음|덮밥|파스타|피자|버거|스테이크|찌개|탕|국밥|우동|라면|면|죽|순두부|냉면|칼국수|돈까스|가츠|초밥|롤))\s+(.+)$"
)


def _normalize_rule_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text or "")
    normalized = "".join(ch for ch in normalized if not ch.isspace())
    normalized = NON_TEXT_RE.sub("", normalized)
    return normalized.casefold()


def _normalized_terms(values: set[str]) -> tuple[str, ...]:
    terms = {_normalize_rule_text(value) for value in values}
    return tuple(sorted((term for term in terms if term), key=len, reverse=True))


SECTION_HEADER_HINTS = {
    _normalize_rule_text(value)
    for value in {
        "사이드",
        "메인",
        "음료",
        "주류",
        "주류/음료",
        "세트",
        "코스",
        "추가",
        "토핑",
        "사리",
        "단품",
        "요리",
        "식사",
        "메뉴",
    }
}
MARKETING_HINTS = _normalized_terms(
    {
        "프리미엄",
        "시그니처",
        "스페셜",
        "베스트",
        "best",
        "추천",
        "인기",
        "대표",
        "고퀄리티",
        "건강",
    }
)
SENTENCE_ENDING_HINTS = _normalized_terms(
    {
        "입니다",
        "습니다",
        "합니다",
        "됩니다",
        "있습니다",
        "가능합니다",
        "드립니다",
    }
)
PARTICIPLE_HINTS = _normalized_terms(
    {
        "들어있",
        "담긴",
        "어우러진",
        "느껴지",
        "숙성",
        "자작하게",
        "곁들인",
        "제공",
        "풍미",
        "식감",
        "특징",
        "위한",
        "맛을 낸",
        "맛을낸",
    }
)
PARTICLE_TOKEN_ENDINGS = ("으로", "에서", "에게", "에는", "와", "과", "의")
KOREAN_DESCRIPTION_TAIL_CUES = _normalized_terms(
    {
        "짭조름",
        "고소",
        "매콤",
        "달콤",
        "바삭",
        "쫄깃",
        "부드러",
        "특제",
        "국내산",
        "계절",
        "조화",
        "풍미",
        "식감",
        "숙성",
        "들어",
        "가득",
        "함께",
        "느껴지",
        "어우러",
    }
)


@dataclass
class MenuCleanCandidate:
    raw_text: str
    cleaned_text: str
    compact_text: str
    status: str
    reasons: List[str] = field(default_factory=list)
    score: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class MenuCleanResult:
    cleaned_items: List[str] = field(default_factory=list)
    kept: List[MenuCleanCandidate] = field(default_factory=list)
    dropped: List[MenuCleanCandidate] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "cleaned_items": list(self.cleaned_items),
            "kept": [item.to_dict() for item in self.kept],
            "dropped": [item.to_dict() for item in self.dropped],
        }


def clean_menu_candidates(candidates: List[str]) -> MenuCleanResult:
    normalized = _normalize_inputs(candidates)
    staged: List[MenuCleanCandidate] = []

    for raw_text in normalized:
        cleaned_text, reasons = _clean_text(raw_text)
        compact_text = _compact_text(cleaned_text)
        metrics = _shape_metrics(cleaned_text)
        score = _quality_score(cleaned_text, metrics)

        candidate = MenuCleanCandidate(
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            compact_text=compact_text,
            status="candidate",
            reasons=reasons,
            score=score,
            metrics=metrics,
        )

        drop_reason = _drop_reason(candidate)
        if drop_reason:
            candidate.status = "dropped"
            candidate.reasons.append(drop_reason)
            staged.append(candidate)
            continue

        candidate.status = "kept"
        staged.append(candidate)

    deduped = _dedupe_candidates(staged)
    resolved = _resolve_overlaps(deduped)
    resolved = _drop_contextual_generics(resolved)

    kept = [item for item in resolved if item.status == "kept"]
    kept.sort(key=lambda item: (-item.score, item.cleaned_text))

    dropped = [item for item in resolved if item.status != "kept"]
    dropped.sort(key=lambda item: (item.raw_text, item.cleaned_text))

    return MenuCleanResult(
        cleaned_items=[item.cleaned_text for item in kept],
        kept=kept,
        dropped=dropped,
    )


def _normalize_inputs(candidates: List[str]) -> List[str]:
    out: List[str] = []
    for value in candidates:
        if not isinstance(value, str):
            continue
        text = " ".join(value.split()).strip()
        if not text:
            continue
        out.append(text)
    return out


def _clean_text(text: str) -> tuple[str, List[str]]:
    current = text
    reasons: List[str] = []

    updated = LEADING_INDEX_RE.sub("", current).strip()
    if updated != current:
        reasons.append("stripped_leading_index")
        current = updated

    updated = EDGE_PUNCT_RE.sub("", current).strip()
    if updated != current:
        reasons.append("trimmed_edge_punctuation")
        current = updated

    updated, normalize_reasons = _normalize_common_menu_text(current)
    if updated != current:
        reasons.extend(normalize_reasons)
        current = updated

    for regex in TRAILING_PRICE_RE_LIST:
        updated = regex.sub("", current).strip()
        if updated != current:
            reasons.append("stripped_trailing_price")
            current = updated
            break

    updated = EDGE_PUNCT_RE.sub("", current).strip()
    if updated != current:
        reasons.append("trimmed_edge_punctuation")
        current = updated

    updated = " ".join(current.split()).strip()
    if updated != current:
        reasons.append("normalized_whitespace")
        current = updated

    return current, reasons


def _drop_reason(candidate: MenuCleanCandidate) -> str:
    text = candidate.cleaned_text
    compact = candidate.compact_text
    metrics = candidate.metrics

    if not text:
        return "empty_after_clean"
    if compact and len(compact) < 2:
        return "too_short"
    if metrics["letter_count"] <= 0:
        return "no_letter_like_chars"
    if metrics["informative_count"] <= 1:
        return "too_little_information"
    if metrics["letter_ratio"] < 0.45 and not _looks_like_numeric_detail_menu_item(text, metrics):
        return "low_letter_ratio"
    if compact in SECTION_HEADER_HINTS:
        return "generic_category_header"
    if _looks_like_description_text(text, compact, metrics):
        return "looks_like_description"
    return ""


def _dedupe_candidates(candidates: List[MenuCleanCandidate]) -> List[MenuCleanCandidate]:
    seen = {}
    out: List[MenuCleanCandidate] = []

    for candidate in candidates:
        if candidate.status != "kept":
            out.append(candidate)
            continue

        key = candidate.compact_text
        if not key:
            candidate.status = "dropped"
            candidate.reasons.append("empty_compact_text")
            out.append(candidate)
            continue

        if key not in seen:
            seen[key] = candidate
            out.append(candidate)
            continue

        incumbent = seen[key]
        better, worse = _choose_better(incumbent, candidate)
        if better is incumbent:
            candidate.status = "dropped"
            candidate.reasons.append("duplicate_lower_quality")
            out.append(candidate)
            continue

        incumbent.status = "dropped"
        incumbent.reasons.append("duplicate_lower_quality")
        seen[key] = candidate

        replaced: List[MenuCleanCandidate] = []
        for item in out:
            if item is incumbent:
                replaced.append(candidate)
            else:
                replaced.append(item)
        out = replaced
        out.append(incumbent)

    return out


def _resolve_overlaps(candidates: List[MenuCleanCandidate]) -> List[MenuCleanCandidate]:
    kept = [item for item in candidates if item.status == "kept"]
    dropped = [item for item in candidates if item.status != "kept"]

    for i, left in enumerate(kept):
        if left.status != "kept" or not left.compact_text:
            continue
        for right in kept[i + 1:]:
            if right.status != "kept" or not right.compact_text:
                continue
            relation = _overlap_relation(left.compact_text, right.compact_text)
            if not relation:
                continue

            better, worse = _choose_better(left, right)
            if better is worse:
                continue
            worse.status = "dropped"
            worse.reasons.append(f"overlap_lower_quality:{relation}")
            dropped.append(worse)

    final_kept = [item for item in kept if item.status == "kept"]
    final_dropped = dropped
    return final_kept + final_dropped


def _drop_contextual_generics(candidates: List[MenuCleanCandidate]) -> List[MenuCleanCandidate]:
    kept = [item for item in candidates if item.status == "kept"]
    common_suffixes = _discover_common_suffix_headers(kept)
    if not common_suffixes:
        return candidates

    for item in kept:
        if item.status != "kept":
            continue

        matched_suffix = _matched_common_suffix(item.compact_text, common_suffixes)
        if not matched_suffix:
            continue

        if item.compact_text == matched_suffix:
            item.status = "dropped"
            item.reasons.append("common_suffix_header")
            continue

        prefix = item.compact_text[: -len(matched_suffix)]
        if _looks_like_repeated_brand(prefix):
            item.status = "dropped"
            item.reasons.append("contextual_brand_title")
            continue

        if _contains_any(item.compact_text, MARKETING_HINTS) and item.metrics["token_count"] <= 3:
            item.status = "dropped"
            item.reasons.append("contextual_marketing_title")

    return candidates


def _discover_common_suffix_headers(candidates: List[MenuCleanCandidate]) -> Dict[str, int]:
    discovered: Dict[str, int] = {}

    for item in candidates:
        compact = item.compact_text
        if not _is_suffix_header_candidate(item):
            continue

        count = 0
        for other in candidates:
            if other is item or len(other.compact_text) <= len(compact) + 1:
                continue
            if _text_ends_with_suffix_variant(other.compact_text, compact):
                count += 1

        if count >= 3:
            discovered[compact] = count

    return discovered


def _is_suffix_header_candidate(candidate: MenuCleanCandidate) -> bool:
    compact = candidate.compact_text
    metrics = candidate.metrics
    if not compact or len(compact) < 2 or len(compact) > 4:
        return False
    if metrics["token_count"] > 1 or metrics["digit_count"] > 0:
        return False
    return True


def _matched_common_suffix(compact_text: str, common_suffixes: Dict[str, int]) -> str:
    for suffix in sorted(common_suffixes.keys(), key=len, reverse=True):
        if len(compact_text) < len(suffix):
            continue
        tail = compact_text[-len(suffix):]
        if _suffix_similarity(tail, suffix) >= _suffix_match_threshold(suffix):
            return suffix
    return ""


def _text_ends_with_suffix_variant(compact_text: str, suffix: str) -> bool:
    if len(compact_text) < len(suffix):
        return False
    return _suffix_similarity(compact_text[-len(suffix):], suffix) >= _suffix_match_threshold(suffix)


def _suffix_similarity(left: str, right: str) -> float:
    if left == right:
        return 1.0
    if len(left) != len(right):
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def _suffix_match_threshold(suffix: str) -> float:
    return 0.5 if len(suffix) <= 2 else 0.66


def _overlap_relation(left: str, right: str) -> str:
    if left == right:
        return "equal"

    left_comp = _comparison_text(left)
    right_comp = _comparison_text(right)
    shorter, longer = (left_comp, right_comp) if len(left_comp) <= len(right_comp) else (right_comp, left_comp)

    if shorter and len(shorter) >= 3 and shorter in longer and (len(shorter) / len(longer)) >= 0.65:
        return "contained"

    if min(len(left_comp), len(right_comp)) >= 3 and SequenceMatcher(None, left_comp, right_comp).ratio() >= 0.82:
        return "near_duplicate"

    return ""


def _comparison_text(compact_text: str) -> str:
    reduced = compact_text
    for term in MARKETING_HINTS:
        reduced = reduced.replace(term, "")
    return reduced or compact_text


def _choose_better(
    left: MenuCleanCandidate,
    right: MenuCleanCandidate,
) -> tuple[MenuCleanCandidate, MenuCleanCandidate]:
    left_key = (round(left.score, 3), -left.metrics["token_count"], -len(left.cleaned_text), -left.metrics["digit_count"])
    right_key = (round(right.score, 3), -right.metrics["token_count"], -len(right.cleaned_text), -right.metrics["digit_count"])
    if left_key >= right_key:
        return left, right
    return right, left


def _quality_score(text: str, metrics: Dict[str, float]) -> float:
    compact = _compact_text(text)
    letter_count = metrics["letter_count"]
    token_count = metrics["token_count"]
    digit_count = metrics["digit_count"]
    symbol_count = metrics["symbol_count"]
    volume_like = bool(VOLUME_LIKE_RE.search(text))

    description_penalty = 6.0 if _looks_like_description_text(text, compact, metrics) else 0.0
    marketing_penalty = 2.0 * _count_terms(compact, MARKETING_HINTS)
    length_penalty = max(0.0, len(text) - 14) * 0.22
    token_penalty = max(0.0, token_count - 2) * 2.0
    digit_penalty = digit_count * (0.35 if volume_like else 0.9)
    symbol_penalty = symbol_count * 0.5
    volume_bonus = 1.5 if volume_like else 0.0

    return float(
        letter_count
        + volume_bonus
        - token_penalty
        - digit_penalty
        - symbol_penalty
        - length_penalty
        - description_penalty
        - marketing_penalty
    )


def _shape_metrics(text: str) -> Dict[str, float]:
    informative = [ch for ch in text if not ch.isspace()]
    letter_count = sum(1 for ch in informative if ch.isalpha())
    digit_count = sum(1 for ch in informative if ch.isdigit())
    symbol_count = max(0, len(informative) - letter_count - digit_count)
    token_count = len(text.split()) if text else 0
    informative_count = len(informative)
    letter_ratio = float(letter_count / informative_count) if informative_count else 0.0
    return {
        "letter_count": float(letter_count),
        "digit_count": float(digit_count),
        "symbol_count": float(symbol_count),
        "token_count": float(token_count),
        "informative_count": float(informative_count),
        "letter_ratio": float(letter_ratio),
    }


def _compact_text(text: str) -> str:
    stripped = "".join(ch for ch in unicodedata.normalize("NFKC", text or "") if not ch.isspace())
    stripped = NON_TEXT_RE.sub("", stripped)
    return stripped.casefold()


def _normalize_common_menu_text(text: str) -> tuple[str, List[str]]:
    current = text
    reasons: List[str] = []

    updated = VOLUME_SUFFIX_RE.sub("ml", current)
    if updated != current:
        current = updated
        reasons.append("normalized_volume_suffix")

    updated = _strip_badge_tokens(current, from_left=True)
    if updated != current:
        current = updated
        reasons.append("stripped_leading_badge")

    updated = _strip_badge_tokens(current, from_left=False)
    if updated != current:
        current = updated
        reasons.append("stripped_trailing_badge")

    updated, trimmed = _trim_korean_mixed_title_tail(current)
    if trimmed:
        current = updated
        reasons.append("trimmed_mixed_description_tail")

    return current, reasons


def _strip_badge_tokens(text: str, from_left: bool) -> str:
    tokens = text.split()
    if not tokens:
        return text

    changed = False
    while tokens:
        idx = 0 if from_left else -1
        token = tokens[idx]
        token_key = BADGE_TOKEN_STRIP_RE.sub("", token).casefold()
        if token_key not in MENU_BADGE_TOKENS:
            break
        if len(tokens) == 1:
            break
        changed = True
        if from_left:
            tokens = tokens[1:]
        else:
            tokens = tokens[:-1]

    if not changed:
        return text
    return " ".join(tokens).strip()


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term and term in text for term in terms)


def _count_terms(text: str, terms: tuple[str, ...]) -> int:
    return sum(1 for term in terms if term and term in text)


def _particle_like_token_count(text: str) -> int:
    count = 0
    for raw_token in text.split():
        token = raw_token.strip(",./|")
        if len(token) < 2:
            continue
        for ending in PARTICLE_TOKEN_ENDINGS:
            if token.endswith(ending):
                count += 1
                break
    return count


def _looks_like_numeric_detail_menu_item(text: str, metrics: Dict[str, float]) -> bool:
    if metrics["letter_count"] < 2:
        return False
    if VOLUME_LIKE_RE.search(text):
        return True
    if ALCOHOL_STRENGTH_RE.search(text):
        return True
    return False


def _trim_korean_mixed_title_tail(text: str) -> tuple[str, bool]:
    if not any("가" <= ch <= "힣" for ch in text):
        return text, False

    match = KOREAN_TITLE_DESCRIPTION_SPLIT_RE.match(text)
    if not match:
        return text, False

    title = match.group(1).strip()
    tail = match.group(2).strip()
    if not title or not tail:
        return text, False

    title_tokens = title.split()
    if len(title_tokens) >= 2 and _looks_like_repeated_brand(_compact_text(title_tokens[0])):
        return text, False

    tail_compact = _compact_text(tail)
    if not tail_compact or len(tail.split()) < 2:
        return text, False

    if _contains_any(tail_compact, KOREAN_DESCRIPTION_TAIL_CUES):
        # 숫자/가격이 붙은 라인은 메뉴명 일부일 가능성이 더 높아 보수적으로 유지한다.
        if re.search(r"(?:원|krw|usd|eur|cny|jpy|元|円|\d)", tail, re.IGNORECASE):
            return text, False
        return title, True

    return text, False


def _looks_like_description_text(text: str, compact_text: str, metrics: Dict[str, float]) -> bool:
    if ", " in text and len(text.split()) >= 4 and not _looks_like_menu_single_comma_conjunction(text, metrics):
        return True
    if _contains_any(compact_text, SENTENCE_ENDING_HINTS):
        return True
    if _contains_any(compact_text, PARTICIPLE_HINTS) and (metrics["token_count"] >= 2 or len(compact_text) >= 10):
        return True
    if _particle_like_token_count(text) >= 2 and metrics["token_count"] >= 3:
        return True
    return False


def _looks_like_menu_single_comma_conjunction(text: str, metrics: Dict[str, float]) -> bool:
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        return False
    if normalized.count(",") != 1:
        return False
    if any(mark in normalized for mark in (".", ";", ":")):
        return False
    if metrics.get("token_count", 0.0) > 10:
        return False
    return bool(SINGLE_COMMA_JOINER_SUFFIX_RE.search(normalized))


def _looks_like_repeated_brand(text: str) -> bool:
    return bool(text and REPEATED_BRAND_RE.fullmatch(text))
