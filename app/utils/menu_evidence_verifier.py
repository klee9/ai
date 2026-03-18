import re
from typing import Iterable, List

from app.agents._0_contracts import AvoidEvidence, RiskItem, RiskSuspect
from app.utils.avoid_ingredient_synonyms import (
    build_canonical_display_map,
    canonicalize_avoid_ingredients,
    find_matching_avoid_canonical,
    get_display_name,
    get_menu_evidence_catalog,
    normalize_ingredient_token,
)


WEAK_INFERENCE_MAX_CONFIDENCE = 0.45
EVIDENCE_RANK = {
    "direct": 4,
    "alias": 3,
    "menu_prior": 2,
    "weak_inference": 1,
    "none": 0,
}


def _contains_term(menu_name: str, term: str) -> bool:
    menu_norm = normalize_ingredient_token(menu_name)
    term_norm = normalize_ingredient_token(term)
    if not menu_norm or not term_norm:
        return False

    if re.fullmatch(r"[a-z0-9 _-]+", term_norm):
        pattern = rf"(?<![a-z0-9]){re.escape(term_norm)}(?![a-z0-9])"
        return re.search(pattern, menu_norm) is not None

    return term_norm in menu_norm


def _match_any(menu_name: str, terms: Iterable[str]) -> str | None:
    for term in terms:
        if _contains_term(menu_name, term):
            return term
    return None


def _verify_suspect(menu_name: str, suspect: RiskSuspect, evidence_catalog: dict) -> RiskSuspect | None:
    canonical = (suspect.canonical or "").strip().casefold()
    if not canonical:
        return None

    entry = evidence_catalog.get(canonical, {})
    direct_terms = entry.get("direct", [])
    strong_terms = entry.get("strong", [])
    prior_terms = entry.get("prior", [])
    direct_hit_menu = _match_any(menu_name, direct_terms)
    alias_hit_menu = _match_any(menu_name, strong_terms)
    prior_hit_menu = _match_any(menu_name, prior_terms)

    evidence_text = (suspect.evidence_text or "").strip() or None
    if evidence_text and not _contains_term(menu_name, evidence_text):
        evidence_text = None

    evidence_type = suspect.evidence_type
    confidence = float(max(0.0, min(1.0, suspect.confidence)))

    if evidence_type == "direct":
        direct_hit = evidence_text and _match_any(evidence_text, direct_terms)
        if not direct_hit:
            return None
        confidence = max(confidence, 0.85)
        return RiskSuspect(
            canonical=canonical,
            evidence_type="direct",
            evidence_text=evidence_text,
            reason=suspect.reason,
            confidence=confidence,
        )

    if evidence_type == "alias":
        alias_hit = evidence_text and _match_any(evidence_text, strong_terms)
        if not alias_hit:
            alias_hit = alias_hit_menu
        if not alias_hit:
            return RiskSuspect(
                canonical=canonical,
                evidence_type="weak_inference",
                evidence_text=None,
                reason=suspect.reason,
                confidence=min(confidence, WEAK_INFERENCE_MAX_CONFIDENCE),
            )
        confidence = max(confidence, 0.75)
        return RiskSuspect(
            canonical=canonical,
            evidence_type="alias",
            evidence_text=evidence_text or alias_hit,
            reason=suspect.reason,
            confidence=confidence,
        )

    if evidence_type == "menu_prior":
        prior_hit = prior_hit_menu
        if not prior_hit:
            return RiskSuspect(
                canonical=canonical,
                evidence_type="weak_inference",
                evidence_text=None,
                reason=suspect.reason,
                confidence=min(confidence, WEAK_INFERENCE_MAX_CONFIDENCE),
            )
        confidence = max(confidence, 0.6)
        return RiskSuspect(
            canonical=canonical,
            evidence_type="menu_prior",
            evidence_text=prior_hit,
            reason=suspect.reason,
            confidence=confidence,
        )

    if evidence_type == "weak_inference":
        if direct_hit_menu:
            return RiskSuspect(
                canonical=canonical,
                evidence_type="direct",
                evidence_text=direct_hit_menu,
                reason=suspect.reason,
                confidence=max(confidence, 0.85),
            )
        if alias_hit_menu:
            return RiskSuspect(
                canonical=canonical,
                evidence_type="alias",
                evidence_text=alias_hit_menu,
                reason=suspect.reason,
                confidence=max(confidence, 0.75),
            )
        if prior_hit_menu:
            return RiskSuspect(
                canonical=canonical,
                evidence_type="menu_prior",
                evidence_text=prior_hit_menu,
                reason=suspect.reason,
                confidence=max(confidence, 0.6),
            )
        return RiskSuspect(
            canonical=canonical,
            evidence_type="weak_inference",
            evidence_text=None,
            reason=suspect.reason,
            confidence=min(confidence, WEAK_INFERENCE_MAX_CONFIDENCE),
        )

    return None


def _detect_strong_menu_canonicals(menu_name: str, evidence_catalog: dict) -> set[str]:
    strong_canonicals: set[str] = set()
    for canonical, entry in evidence_catalog.items():
        if _match_any(menu_name, entry.get("direct", [])):
            strong_canonicals.add(canonical)
            continue
        if _match_any(menu_name, entry.get("strong", [])):
            strong_canonicals.add(canonical)
            continue
        if _match_any(menu_name, entry.get("prior", [])):
            strong_canonicals.add(canonical)
    return strong_canonicals


def _resolve_canonical_conflicts(
    menu_name: str,
    verified_pairs: List[tuple[str, RiskSuspect]],
    evidence_catalog: dict,
) -> List[tuple[str, RiskSuspect]]:
    if not verified_pairs:
        return []

    strong_menu_canonicals = _detect_strong_menu_canonicals(menu_name, evidence_catalog)
    if not strong_menu_canonicals:
        return verified_pairs

    filtered: List[tuple[str, RiskSuspect]] = []
    for matched_canonical, suspect in verified_pairs:
        if (
            suspect.evidence_type == "weak_inference"
            and suspect.canonical not in strong_menu_canonicals
        ):
            continue
        filtered.append((matched_canonical, suspect))
    return filtered


def _merge_verified_suspects(
    verified_pairs: List[tuple[str, RiskSuspect]],
) -> List[tuple[str, RiskSuspect]]:
    best_by_key: dict[tuple[str, str], tuple[str, RiskSuspect]] = {}
    for matched_canonical, suspect in verified_pairs:
        key = (matched_canonical, suspect.canonical)
        current = best_by_key.get(key)
        if current is None:
            best_by_key[key] = (matched_canonical, suspect)
            continue

        _, current_suspect = current
        current_rank = EVIDENCE_RANK.get(current_suspect.evidence_type, 0)
        new_rank = EVIDENCE_RANK.get(suspect.evidence_type, 0)
        if new_rank > current_rank or (
            new_rank == current_rank and suspect.confidence > current_suspect.confidence
        ):
            best_by_key[key] = (matched_canonical, suspect)

    return sorted(
        best_by_key.values(),
        key=lambda pair: (
            EVIDENCE_RANK.get(pair[1].evidence_type, 0),
            pair[1].confidence,
        ),
        reverse=True,
    )


def verify_risk_items(risk_items: List[RiskItem], avoid_terms: List[str], lang: str = "ko") -> List[RiskItem]:
    if not risk_items:
        return []

    evidence_catalog = get_menu_evidence_catalog()
    allowed_canonicals = {
        canonical.casefold()
        for canonical in canonicalize_avoid_ingredients(avoid_terms)
        if isinstance(canonical, str) and canonical.strip()
    }
    display_by_requested_canonical = build_canonical_display_map(avoid_terms, lang=lang)

    verified_items: List[RiskItem] = []
    for item in risk_items:
        if not isinstance(item, RiskItem):
            continue

        verified_pairs: List[tuple[str, RiskSuspect]] = []
        for suspect in item.suspects:
            matched_avoid_canonical = find_matching_avoid_canonical(suspect.canonical, allowed_canonicals)
            if matched_avoid_canonical is None:
                continue
            verified = _verify_suspect(item.menu, suspect, evidence_catalog)
            if verified is not None:
                verified_pairs.append((matched_avoid_canonical, verified))

        verified_pairs = _merge_verified_suspects(verified_pairs)
        verified_pairs = _resolve_canonical_conflicts(item.menu, verified_pairs, evidence_catalog)
        verified_suspects = [suspect for _, suspect in verified_pairs]

        matched_avoid: List[str] = []
        avoid_evidence: List[AvoidEvidence] = []
        suspected_ingredients: List[str] = []
        for matched_avoid_canonical, suspect in verified_pairs:
            display_name = display_by_requested_canonical.get(
                matched_avoid_canonical,
                get_display_name(matched_avoid_canonical, lang=lang),
            )
            if display_name not in matched_avoid:
                matched_avoid.append(display_name)
            if display_name not in suspected_ingredients:
                suspected_ingredients.append(display_name)
            avoid_evidence.append(
                AvoidEvidence(
                    ingredient=display_name,
                    canonical=suspect.canonical,
                    evidence_type=suspect.evidence_type,
                    evidence_text=suspect.evidence_text,
                    reason=suspect.reason,
                    confidence=suspect.confidence,
                )
            )

        item_confidence = item.confidence
        if verified_suspects:
            item_confidence = max(item_confidence, max(s.confidence for s in verified_suspects))

        verified_items.append(
            RiskItem(
                menu=item.menu,
                risk=0,
                confidence=item_confidence,
                suspected_ingredients=suspected_ingredients,
                suspects=verified_suspects,
                matched_avoid=matched_avoid,
                avoid_evidence=avoid_evidence,
            )
        )

    return verified_items
