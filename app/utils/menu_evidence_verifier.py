import re
from typing import Iterable, List

from app.agents._0_contracts import AvoidEvidence, RiskItem, RiskSuspect
from app.utils.avoid_ingredient_synonyms import (
    build_canonical_display_map,
    canonicalize_avoid_ingredients,
    find_matching_avoid_canonical,
    get_canonical_ingredient,
    get_canonical_ancestors,
    get_display_name,
    get_menu_evidence_catalog,
    normalize_ingredient_token,
)


WEAK_INFERENCE_MAX_CONFIDENCE = 0.45
RELATION_FALLBACK_MIN_CONFIDENCE = 0.6
SOFT_FALLBACK_MIN_INPUT_CONFIDENCE = 0.35
SOFT_FALLBACK_SCALE = 0.45
SOFT_FALLBACK_MAX_CONFIDENCE = 0.25
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
    weak_terms = entry.get("weak", [])
    direct_hit_menu = _match_any(menu_name, direct_terms)
    alias_hit_menu = _match_any(menu_name, strong_terms)
    prior_hit_menu = _match_any(menu_name, prior_terms)
    weak_hit_menu = _match_any(menu_name, weak_terms)

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
            if not weak_hit_menu:
                return None
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
        prior_hit = prior_hit_menu
        if not prior_hit:
            if not weak_hit_menu:
                return None
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
        if weak_hit_menu:
            return RiskSuspect(
                canonical=canonical,
                evidence_type="weak_inference",
                evidence_text=weak_hit_menu,
                reason=suspect.reason,
                confidence=min(confidence, WEAK_INFERENCE_MAX_CONFIDENCE),
            )
        return None

    return None


def _canonical_match_relation(source_canonical: str, target_canonical: str) -> str:
    source = (source_canonical or "").strip().casefold()
    target = (target_canonical or "").strip().casefold()
    if not source or not target:
        return "none"
    if source == target:
        return "exact"

    source_ancestors = set(get_canonical_ancestors(source))
    target_ancestors = set(get_canonical_ancestors(target))
    if target in source_ancestors:
        return "child_to_parent"
    if source in target_ancestors:
        return "parent_to_child"
    if source_ancestors & target_ancestors:
        return "sibling_family"
    return "none"


def _has_canonical_menu_signal(menu_name: str, canonical: str, evidence_catalog: dict) -> bool:
    canonical_norm = (canonical or "").strip().casefold()
    if not canonical_norm:
        return False

    entry = evidence_catalog.get(canonical_norm, {})
    terms: List[str] = []
    for section in ("direct", "strong", "prior"):
        values = entry.get(section, [])
        if isinstance(values, list):
            terms.extend(value for value in values if isinstance(value, str))
    return _match_any(menu_name, terms) is not None


def _has_relation_menu_signal(
    menu_name: str,
    source_canonical: str,
    target_canonical: str,
    evidence_catalog: dict,
) -> bool:
    candidates = [source_canonical, target_canonical]
    candidates.extend(get_canonical_ancestors(source_canonical))
    candidates.extend(get_canonical_ancestors(target_canonical))

    seen = set()
    for candidate in candidates:
        candidate_norm = (candidate or "").strip().casefold()
        if not candidate_norm or candidate_norm in seen:
            continue
        seen.add(candidate_norm)
        if _has_canonical_menu_signal(menu_name, candidate_norm, evidence_catalog):
            return True
    return False


def _fallback_verify_by_relation(
    menu_name: str,
    suspect: RiskSuspect,
    relation: str,
    matched_canonical: str,
    evidence_catalog: dict,
) -> RiskSuspect | None:
    if relation not in {"child_to_parent", "parent_to_child", "sibling_family"}:
        return None
    if suspect.evidence_type != "menu_prior":
        return None
    if float(suspect.confidence) < RELATION_FALLBACK_MIN_CONFIDENCE:
        return None

    source_canonical = (suspect.canonical or "").strip().casefold()
    target_canonical = (matched_canonical or "").strip().casefold()
    if not _has_relation_menu_signal(menu_name, source_canonical, target_canonical, evidence_catalog):
        return None

    relation_reason = {
        "child_to_parent": "ingredient-family linked match (child->parent)",
        "parent_to_child": "ingredient-family linked match (parent->child)",
        "sibling_family": "ingredient-family linked match (sibling)",
    }.get(relation, "ingredient-family linked match")
    base_reason = (suspect.reason or "").strip()
    reason = f"{base_reason} ({relation_reason})" if base_reason else relation_reason

    return RiskSuspect(
        canonical=source_canonical,
        evidence_type="weak_inference",
        evidence_text=None,
        reason=reason,
        confidence=min(WEAK_INFERENCE_MAX_CONFIDENCE, max(0.35, suspect.confidence)),
    )


def _soft_fallback_verify(suspect: RiskSuspect) -> RiskSuspect | None:
    """
    메뉴 텍스트 신호가 부족해 strict verification을 통과하지 못해도,
    LLM이 준 중간 이상 confidence는 낮은 가중치 weak 추론으로 보존한다.
    """
    canonical = (suspect.canonical or "").strip().casefold()
    if not canonical:
        return None

    base_conf = float(max(0.0, min(1.0, suspect.confidence)))
    if base_conf < SOFT_FALLBACK_MIN_INPUT_CONFIDENCE:
        return None

    softened_conf = min(
        SOFT_FALLBACK_MAX_CONFIDENCE,
        max(0.08, base_conf * SOFT_FALLBACK_SCALE),
    )
    reason = (suspect.reason or "").strip()
    if reason:
        reason = f"{reason} (soft fallback)"
    else:
        reason = "soft fallback from model prior"

    return RiskSuspect(
        canonical=canonical,
        evidence_type="weak_inference",
        evidence_text=None,
        reason=reason,
        confidence=softened_conf,
    )


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
            # 이전에는 exact 약한 추론을 제거했지만, soft calibration 정책에서는
            # 낮은 confidence로 보존해 점수 단계에서 확률적으로 반영한다.
            matched_norm = (matched_canonical or "").strip().casefold()
            suspect_norm = (suspect.canonical or "").strip().casefold()
            if matched_norm == suspect_norm:
                filtered.append(
                    (
                        matched_canonical,
                        RiskSuspect(
                            canonical=suspect.canonical,
                            evidence_type="weak_inference",
                            evidence_text=None,
                            reason=suspect.reason,
                            confidence=min(suspect.confidence, SOFT_FALLBACK_MAX_CONFIDENCE),
                        ),
                    )
                )
                continue
        filtered.append((matched_canonical, suspect))
    return filtered


def _infer_menu_ingredient_canonicals(menu_name: str, evidence_catalog: dict) -> List[str]:
    menu_norm = normalize_ingredient_token(menu_name)
    if not menu_norm:
        return []

    ranked_hits: List[tuple[int, int, str]] = []
    for canonical, entry in evidence_catalog.items():
        matched_term = _match_any(menu_name, entry.get("strong", []))
        rank = 3
        if not matched_term:
            matched_term = _match_any(menu_name, entry.get("prior", []))
            rank = 2
        if not matched_term:
            direct_terms = [
                term
                for term in entry.get("direct", [])
                if len(normalize_ingredient_token(term)) >= 2
            ]
            matched_term = _match_any(menu_name, direct_terms)
            rank = 1
        if not matched_term:
            continue

        term_norm = normalize_ingredient_token(matched_term)
        position = menu_norm.find(term_norm) if term_norm else -1
        if position < 0:
            position = 10**6
        ranked_hits.append((rank, position, canonical))

    ranked_hits.sort(key=lambda item: (-item[0], item[1], item[2]))

    ordered_canonicals: List[str] = []
    seen = set()
    for _, _, canonical in ranked_hits:
        for candidate in [canonical, *get_canonical_ancestors(canonical)]:
            candidate_norm = (candidate or "").strip().casefold()
            if not candidate_norm or candidate_norm in seen:
                continue
            seen.add(candidate_norm)
            ordered_canonicals.append(candidate_norm)
    return ordered_canonicals


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


def _infer_fallback_pairs_from_menu(
    menu_name: str,
    allowed_canonicals: set[str],
    evidence_catalog: dict,
    matched_avoid_already: set[str],
) -> List[tuple[str, RiskSuspect]]:
    inferred_canonicals = _infer_menu_ingredient_canonicals(menu_name, evidence_catalog)
    out: List[tuple[str, RiskSuspect]] = []

    for inferred in inferred_canonicals:
        matched_avoid = find_matching_avoid_canonical(inferred, allowed_canonicals)
        if matched_avoid is None:
            continue
        if matched_avoid in matched_avoid_already:
            continue

        entry = evidence_catalog.get(inferred, {})
        direct_hit = _match_any(menu_name, entry.get("direct", []))
        strong_hit = _match_any(menu_name, entry.get("strong", []))
        prior_hit = _match_any(menu_name, entry.get("prior", []))

        evidence_type = "weak_inference"
        evidence_text = None
        confidence = 0.25

        if direct_hit:
            evidence_type = "direct"
            evidence_text = direct_hit
            confidence = 0.85
        elif strong_hit:
            evidence_type = "alias"
            evidence_text = strong_hit
            confidence = 0.75
        elif prior_hit:
            evidence_type = "menu_prior"
            evidence_text = prior_hit
            confidence = 0.60

        out.append(
            (
                matched_avoid,
                RiskSuspect(
                    canonical=inferred,
                    evidence_type=evidence_type,
                    evidence_text=evidence_text,
                    reason="menu signal fallback",
                    confidence=confidence,
                ),
            )
        )

    return out


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
        menu_name = (item.menu_original or item.menu or "").strip()
        if not menu_name:
            menu_name = item.menu

        verified_pairs: List[tuple[str, RiskSuspect]] = []
        for suspect in item.suspects:
            matched_avoid_canonical = find_matching_avoid_canonical(suspect.canonical, allowed_canonicals)
            if matched_avoid_canonical is None:
                continue
            relation = _canonical_match_relation(suspect.canonical, matched_avoid_canonical)
            verified = _verify_suspect(menu_name, suspect, evidence_catalog)
            if verified is None:
                verified = _fallback_verify_by_relation(
                    menu_name,
                    suspect,
                    relation,
                    matched_avoid_canonical,
                    evidence_catalog,
                )
            if verified is None:
                verified = _soft_fallback_verify(suspect)
            if verified is not None:
                verified_pairs.append((matched_avoid_canonical, verified))

        # LLM이 suspect를 누락해도 메뉴명 자체에 강한 시그널(예: 스테이크, 제육)이 있으면
        # 기피 재료 매칭 근거를 보수적으로 보강한다.
        matched_avoid_already = {
            matched
            for matched, _ in verified_pairs
            if isinstance(matched, str) and matched.strip()
        }
        verified_pairs.extend(
            _infer_fallback_pairs_from_menu(
                menu_name=menu_name,
                allowed_canonicals=allowed_canonicals,
                evidence_catalog=evidence_catalog,
                matched_avoid_already=matched_avoid_already,
            )
        )

        verified_pairs = _merge_verified_suspects(verified_pairs)
        verified_pairs = _resolve_canonical_conflicts(menu_name, verified_pairs, evidence_catalog)
        verified_suspects = [suspect for _, suspect in verified_pairs]

        matched_avoid: List[str] = []
        avoid_evidence: List[AvoidEvidence] = []
        suspected_ingredients: List[str] = []
        for raw_suspected in item.suspected_ingredients:
            if not isinstance(raw_suspected, str):
                continue
            cleaned = raw_suspected.strip()
            if not cleaned:
                continue
            canonical = get_canonical_ingredient(cleaned, mode="input") or cleaned.casefold()
            display_name = get_display_name(canonical, lang=lang) if canonical in evidence_catalog else cleaned
            if display_name not in suspected_ingredients:
                suspected_ingredients.append(display_name)
        inferred_menu_canonicals = _infer_menu_ingredient_canonicals(menu_name, evidence_catalog)
        for canonical in inferred_menu_canonicals:
            display_name = display_by_requested_canonical.get(
                canonical,
                get_display_name(canonical, lang=lang),
            )
            if display_name not in suspected_ingredients:
                suspected_ingredients.append(display_name)
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
                menu=menu_name,
                menu_original=menu_name,
                risk=0,
                confidence=item_confidence,
                suspected_ingredients=suspected_ingredients,
                suspects=verified_suspects,
                matched_avoid=matched_avoid,
                avoid_evidence=avoid_evidence,
            )
        )

    return verified_items
