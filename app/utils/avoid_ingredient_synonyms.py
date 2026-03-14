import re
import unicodedata
from typing import Dict, Iterable, List


def normalize_ingredient_token(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    normalized = "".join(
        ch for ch in unicodedata.normalize("NFKD", normalized) if not unicodedata.combining(ch)
    )
    return normalized


# 사용자 입력을 대표 재료명으로 정규화하기 위한 간단한 동의어 사전.
# 필요 시 도메인 지식을 계속 추가할 수 있도록 canonical -> variants 구조로 유지한다.
AVOID_INGREDIENT_SYNONYMS: Dict[str, List[str]] = {
    "egg": ["eggs", "계란", "달걀", "계란류", "달걀류", "난류", "계란흰자", "계란 노른자"],
    "milk": ["우유", "젖", "유제품", "유당", "락토스", "milk dairy", "dairy"],
    "cheese": ["치즈", "mozzarella", "mozzarella cheese", "parmesan", "cream cheese"],
    "butter": ["버터"],
    "peanut": ["peanuts", "땅콩", "땅콩류"],
    "tree nut": [
        "tree nuts",
        "견과류",
        "nut",
        "nuts",
        "아몬드",
        "호두",
        "캐슈넛",
        "피스타치오",
        "헤이즐넛",
    ],
    "soy": ["soybean", "soybeans", "대두", "콩", "두유", "soy milk"],
    "wheat": ["밀", "밀가루", "소맥", "글루텐", "gluten", "부침가루", "튀김가루"],
    "shrimp": ["새우", "크릴", "shrimp paste"],
    "crab": ["게", "crab meat"],
    "shellfish": ["조개", "조개류", "패류", "굴", "홍합", "전복", "가리비"],
    "fish": ["생선", "어류", "멸치", "참치", "연어", "fish sauce"],
    "sesame": ["참깨", "깨", "sesame seed", "sesame seeds", "tahini"],
    "beef": ["소고기", "쇠고기"],
    "pork": ["돼지고기", "돈육", "삼겹살", "베이컨", "햄"],
    "chicken": ["닭", "닭고기", "계육"],
}


def build_avoid_synonym_lookup() -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical, variants in AVOID_INGREDIENT_SYNONYMS.items():
        normalized_canonical = normalize_ingredient_token(canonical)
        if normalized_canonical:
            lookup[normalized_canonical] = canonical
        for variant in variants:
            normalized_variant = normalize_ingredient_token(variant)
            if normalized_variant:
                lookup[normalized_variant] = canonical
    return lookup


def canonicalize_avoid_ingredients(ingredients: Iterable[str]) -> List[str]:
    lookup = build_avoid_synonym_lookup()
    canonicalized: List[str] = []
    seen = set()

    for ingredient in ingredients:
        normalized = normalize_ingredient_token(ingredient)
        if not normalized:
            continue
        canonical = lookup.get(normalized, re.sub(r"\s+", " ", ingredient).strip())
        if canonical and canonical not in seen:
            seen.add(canonical)
            canonicalized.append(canonical)

    return canonicalized


def build_avoid_lookup(ingredients: Iterable[str]) -> Dict[str, str]:
    synonym_lookup = build_avoid_synonym_lookup()
    avoid_lookup: Dict[str, str] = {}

    for ingredient in ingredients:
        canonical = synonym_lookup.get(normalize_ingredient_token(ingredient), ingredient)
        for token, mapped in synonym_lookup.items():
            if mapped == canonical:
                avoid_lookup[token] = canonical

        normalized_canonical = normalize_ingredient_token(canonical)
        if normalized_canonical:
            avoid_lookup[normalized_canonical] = canonical

        normalized_original = normalize_ingredient_token(ingredient)
        if normalized_original:
            avoid_lookup[normalized_original] = canonical

    return avoid_lookup
