import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Dict, List


LEADING_INDEX_RE = re.compile(
    r"^\s*(?:[\(\[]?\d{1,3}[\)\].:,-]?\s+|[A-Za-z][\).:-]\s+|[ivxlcdm]+\.\s+)",
    re.IGNORECASE,
)
TRAILING_PRICE_RE_LIST = [
    re.compile(r"\s+[$€£¥₩]\s*\d+(?:[.,]\d{1,2})?$"),
    re.compile(r"\s+\d+(?:[.,]\d{1,2})?\s*(?:원|krw|usd|eur|cny|jpy|元|円)$", re.IGNORECASE),
    re.compile(r"\s+\d{1,3}(?:[.,]\d{1,2})$"),
    re.compile(r"(?<=\D)\d{1,3}(?:[.,]\d{1,2})$"),
]
EDGE_PUNCT_RE = re.compile(r"^[\s\-–—:;,.|/\\•·*]+|[\s\-–—:;,.|/\\•·*]+$")
NON_TEXT_RE = re.compile(r"[\s\-_–—:;,.|/\\•·*()\[\]{}<>]+")


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
    metrics = candidate.metrics

    if not text:
        return "empty_after_clean"
    if candidate.compact_text and len(candidate.compact_text) < 2:
        return "too_short"
    if metrics["letter_count"] <= 0:
        return "no_letter_like_chars"
    if metrics["informative_count"] <= 1:
        return "too_little_information"
    if metrics["letter_ratio"] < 0.45:
        return "low_letter_ratio"
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


def _overlap_relation(left: str, right: str) -> str:
    if left == right:
        return "equal"
    shorter, longer = (left, right) if len(left) <= len(right) else (right, left)
    if not shorter or len(shorter) < 3:
        return ""
    if shorter in longer and (len(shorter) / len(longer)) >= 0.65:
        return "contained"
    return ""


def _choose_better(
    left: MenuCleanCandidate,
    right: MenuCleanCandidate,
) -> tuple[MenuCleanCandidate, MenuCleanCandidate]:
    left_key = (round(left.score, 3), -left.metrics["token_count"], -left.metrics["digit_count"], -len(left.cleaned_text))
    right_key = (round(right.score, 3), -right.metrics["token_count"], -right.metrics["digit_count"], -len(right.cleaned_text))
    if left_key >= right_key:
        return left, right
    return right, left


def _quality_score(text: str, metrics: Dict[str, float]) -> float:
    letter_count = metrics["letter_count"]
    token_count = metrics["token_count"]
    digit_count = metrics["digit_count"]
    symbol_count = metrics["symbol_count"]
    length_penalty = max(0.0, len(text) - 18) * 0.12
    token_penalty = max(0.0, token_count - 3) * 1.4
    digit_penalty = digit_count * 0.9
    symbol_penalty = symbol_count * 0.5
    return float(letter_count - token_penalty - digit_penalty - symbol_penalty - length_penalty)


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
