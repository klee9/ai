import re
import unicodedata
from typing import Dict, Iterable, List, Literal, Optional


AliasMode = Literal["input", "menu_strong", "menu_all"]
SUPPORTED_LOCALIZED_LANGS = ("ko", "en", "es")


def normalize_ingredient_token(text: str) -> str:
    if not isinstance(text, str):
        return ""
    normalized = re.sub(r"\s+", " ", text).strip().lower()
    normalized = "".join(
        ch for ch in unicodedata.normalize("NFKD", normalized) if not unicodedata.combining(ch)
    )
    return normalized


def _iter_supported_localized_terms(values) -> Iterable[str]:
    if not isinstance(values, dict):
        return
    for lang in SUPPORTED_LOCALIZED_LANGS:
        localized_terms = values.get(lang)
        # Spanish fallback: if es metadata is missing, reuse en terms.
        if localized_terms is None and lang == "es":
            localized_terms = values.get("en")

        if isinstance(localized_terms, str):
            if localized_terms.strip():
                yield localized_terms
            continue
        if isinstance(localized_terms, list):
            for term in localized_terms:
                if isinstance(term, str) and term.strip():
                    yield term


# catalog 구조 원칙:
# - display: 사용자에게 보여줄 표준명
# - input_aliases: 사용자가 자유형 입력으로 말할 수 있는 표현(비교적 넓게)
# - menu_evidence_aliases: 메뉴명에서 strong alias/direct 검증에 쓸 토큰
# - menu_prior_aliases: 메뉴명 자체가 특정 재료와 강하게 연결되는 대표 음식명
# - menu_weak_aliases: 메뉴명에서 약한 prior 정도로만 참고할 토큰
AVOID_INGREDIENT_CATALOG: Dict[str, Dict[str, object]] = {
    "egg": {
        "display": {"ko": "계란", "en": "egg"},
        "input_aliases": {
            "ko": ["계란", "달걀", "난류", "계란류", "달걀류", "계란흰자", "계란 노른자", "계란말이", "지단", "수란", "후라이"],
            "en": ["egg", "eggs", "egg white", "egg yolk", "omelet", "scrambled egg", "poached egg", "fried egg"],
        },
        "menu_evidence_aliases": {
            "ko": ["계란", "달걀", "계란흰자", "계란 노른자"],
            "en": ["egg", "eggs", "egg white", "egg yolk"],
        },
        "menu_weak_aliases": {
            "ko": ["계란말이", "지단", "수란", "후라이"],
            "en": ["omelet", "scrambled egg", "poached egg", "fried egg"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
    },
    "dairy": {
        "display": {"ko": "유제품", "en": "dairy"},
        "input_aliases": {
            "ko": ["유제품", "우유류", "乳製品"],
            "en": ["dairy", "dairy product", "dairy products"],
        },
        "menu_evidence_aliases": {"ko": [], "en": []},
        "menu_prior_aliases": {
            "ko": ["알프레도", "크림파스타", "크림소스", "라자냐", "피자", "마르게리타", "말라이"],
            "en": ["alfredo", "cream pasta", "cream sauce", "lasagna", "pizza", "margherita", "malai"],
        },
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "milk": {
        "display": {"ko": "우유", "en": "milk"},
        "parents": ["dairy"],
        "input_aliases": {
            "ko": ["우유", "젖", "유당", "락토스", "연유", "분유"],
            "en": ["milk", "whole milk", "skim milk", "condensed milk", "lactose"],
        },
        "menu_evidence_aliases": {
            "ko": ["우유", "연유", "분유"],
            "en": ["milk", "condensed milk"],
        },
        "menu_weak_aliases": {
            "ko": ["생크림"],
            "en": ["cream"],
        },
        "menu_prior_aliases": {
            "ko": ["밀크티", "밀크쉐이크", "말라이"],
            "en": ["milk tea", "milkshake", "malai"],
        },
    },
    "cheese": {
        "display": {"ko": "치즈", "en": "cheese"},
        "parents": ["dairy"],
        "input_aliases": {
            "ko": ["치즈", "모짜렐라", "크림치즈", "파르메산", "체다치즈"],
            "en": ["cheese", "mozzarella", "cream cheese", "parmesan", "cheddar"],
        },
        "menu_evidence_aliases": {
            "ko": ["치즈", "모짜렐라", "크림치즈", "파르메산", "체다치즈"],
            "en": ["cheese", "mozzarella", "cream cheese", "parmesan", "cheddar"],
        },
        "menu_prior_aliases": {
            "ko": ["피자", "라자냐", "알프레도", "마르게리타"],
            "en": ["pizza", "lasagna", "alfredo", "margherita", "queso"],
        },
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "butter": {
        "display": {"ko": "버터", "en": "butter"},
        "parents": ["dairy"],
        "input_aliases": {
            "ko": ["버터", "가염버터", "무염버터"],
            "en": ["butter", "salted butter", "unsalted butter"],
        },
        "menu_evidence_aliases": {
            "ko": ["버터", "가염버터", "무염버터"],
            "en": ["butter", "salted butter", "unsalted butter"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "peanut": {
        "display": {"ko": "땅콩", "en": "peanut"},
        "input_aliases": {
            "ko": ["땅콩", "땅콩류", "땅콩버터"],
            "en": ["peanut", "peanuts", "peanut butter"],
        },
        "menu_evidence_aliases": {
            "ko": ["땅콩", "땅콩버터"],
            "en": ["peanut", "peanuts", "peanut butter"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "tree nut": {
        "display": {"ko": "견과류", "en": "tree nut"},
        "input_aliases": {
            "ko": ["견과류", "아몬드", "호두", "캐슈넛", "피스타치오", "헤이즐넛", "잣", "밤", "마카다미아"],
            "en": ["tree nut", "tree nuts", "almond", "walnut", "cashew", "pistachio", "hazelnut", "pine nut", "macadamia"],
        },
        "menu_evidence_aliases": {
            "ko": ["아몬드", "호두", "캐슈넛", "피스타치오", "헤이즐넛", "잣", "밤", "마카다미아"],
            "en": ["almond", "walnut", "cashew", "pistachio", "hazelnut", "pine nut", "macadamia"],
        },
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "soy": {
        "display": {"ko": "대두", "en": "soy"},
        "input_aliases": {
            "ko": ["대두", "콩", "콩류", "두부", "유부", "두유", "된장", "간장", "미소", "콩국", "콩국수"],
            "en": ["soy", "soybean", "soybeans", "tofu", "fried tofu", "soy milk", "miso", "soy sauce"],
        },
        "menu_evidence_aliases": {
            "ko": ["대두", "콩", "두부", "유부", "콩국", "콩국수", "된장"],
            "en": ["soy", "soybean", "soybeans", "tofu", "fried tofu", "soy milk", "bean curd"],
        },
        "menu_weak_aliases": {
            "ko": ["간장", "미소"],
            "en": ["miso", "soy sauce"],
        },
        "menu_prior_aliases": {"ko": ["된장찌개"], "en": []},
    },
    "gluten": {
        "display": {"ko": "글루텐", "en": "gluten"},
        "input_aliases": {
            "ko": ["글루텐"],
            "en": ["gluten"],
        },
        "menu_evidence_aliases": {
            "ko": ["글루텐"],
            "en": ["gluten"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "wheat": {
        "display": {"ko": "밀", "en": "wheat"},
        "parents": ["gluten"],
        "input_aliases": {
            "ko": ["밀", "밀가루", "소맥", "글루텐", "면", "국수", "칼국수", "우동", "라면", "파스타", "수제비", "빵", "토스트", "만두피", "튀김옷"],
            "en": ["wheat", "flour", "gluten", "noodle", "noodles", "pasta", "udon", "ramen", "bread", "toast", "dumpling wrapper", "batter"],
        },
        "menu_evidence_aliases": {
            "ko": ["밀", "밀가루", "소맥", "글루텐"],
            "en": ["wheat", "flour", "gluten", "wheat flour"],
        },
        "menu_weak_aliases": {
            "ko": ["면", "국수", "칼국수", "우동", "라면", "파스타", "수제비", "빵", "토스트", "만두피", "튀김옷"],
            "en": ["noodle", "noodles", "pasta", "udon", "ramen", "bread", "toast", "dumpling wrapper", "batter"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
    },
    "shrimp": {
        "display": {"ko": "새우", "en": "shrimp"},
        "input_aliases": {
            "ko": ["새우", "새우살", "건새우", "새우젓", "크릴", "쉬림프"],
            "en": ["shrimp", "prawn", "dried shrimp", "shrimp paste", "shrimp ball", "krill"],
        },
        "menu_evidence_aliases": {
            "ko": ["새우", "새우살", "건새우", "새우젓"],
            "en": ["shrimp", "prawn", "dried shrimp", "shrimp paste", "shrimp ball"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "crab": {
        "display": {"ko": "게", "en": "crab"},
        "input_aliases": {
            "ko": ["게", "꽃게", "대게", "게살", "게장", "간장게장", "양념게장"],
            "en": ["crab", "crab meat", "blue crab", "snow crab", "soy-marinated crab"],
        },
        "menu_evidence_aliases": {
            "ko": ["게", "꽃게", "대게", "게살", "게장", "간장게장", "양념게장"],
            "en": ["crab", "crab meat", "blue crab", "snow crab"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "shellfish": {
        "display": {"ko": "패류", "en": "shellfish"},
        "input_aliases": {
            "ko": ["패류", "조개", "조개류", "굴", "홍합", "전복", "가리비", "바지락", "꼬막", "소라", "모시조개", "키조개", "재첩"],
            "en": ["shellfish", "clam", "clams", "oyster", "mussel", "abalone", "scallop", "cockle", "whelk"],
        },
        "menu_evidence_aliases": {
            "ko": ["조개", "굴", "홍합", "전복", "가리비", "바지락", "꼬막", "소라", "모시조개", "키조개", "재첩"],
            "en": ["clam", "clams", "oyster", "mussel", "abalone", "scallop", "cockle", "whelk"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
        "menu_weak_aliases": {"ko": [], "en": []},
    },
    "fish": {
        "display": {"ko": "생선", "en": "fish"},
        "input_aliases": {
            "ko": ["생선", "어류", "멸치", "참치", "연어", "고등어", "명태", "동태", "북어", "황태", "코다리", "대구", "갈치", "삼치", "꽁치", "도미", "참도미", "광어", "우럭", "방어", "가자미", "도다리", "민어", "농어", "장어", "민물장어", "붕장어", "아나고", "복어", "메로", "아귀", "임연수", "전어", "청어", "병어", "가리비", "액젓", "생선육수", "회", "생선회", "어묵", "생선까스", "가쓰오", "가다랑어", "멸치육수"],
            "en": ["fish", "anchovy", "tuna", "salmon", "mackerel", "pollock", "cod", "dried pollock", "yellow pollock", "sea bream", "snapper", "halibut", "flounder", "rockfish", "yellowtail", "eel", "conger eel", "pufferfish", "anglerfish", "saury", "hairtail", "fish sauce", "fish stock", "sashimi", "fish cake", "fish cutlet", "bonito"],
        },
        "menu_evidence_aliases": {
            "ko": ["생선", "어류", "멸치", "참치", "연어", "고등어", "명태", "동태", "북어", "황태", "코다리", "대구", "갈치", "삼치", "꽁치", "도미", "참도미", "광어", "우럭", "방어", "가자미", "도다리", "민어", "농어", "장어", "민물장어", "붕장어", "아나고", "복어", "메로", "아귀", "임연수", "전어", "청어", "병어", "회", "생선회", "어묵", "생선까스", "가쓰오", "가다랑어"],
            "en": ["fish", "anchovy", "tuna", "salmon", "mackerel", "pollock", "cod", "sea bream", "snapper", "halibut", "flounder", "rockfish", "yellowtail", "eel", "conger eel", "pufferfish", "anglerfish", "sashimi", "fish cake", "fish cutlet", "bonito"],
        },
        "menu_weak_aliases": {
            "ko": ["액젓", "생선육수", "멸치육수"],
            "en": ["fish sauce", "fish stock"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
    },
    "sesame": {
        "display": {"ko": "참깨", "en": "sesame"},
        "input_aliases": {
            "ko": ["참깨", "참기름", "깨소금", "볶음참깨"],
            "en": ["sesame", "sesame seed", "sesame seeds", "sesame oil", "tahini"],
        },
        "menu_evidence_aliases": {
            "ko": ["참깨", "볶음참깨"],
            "en": ["sesame", "sesame seed", "sesame seeds"],
        },
        "menu_weak_aliases": {
            "ko": ["참기름", "깨소금"],
            "en": ["sesame oil", "tahini"],
        },
        "menu_prior_aliases": {"ko": [], "en": []},
    },
    "beef": {
        "display": {"ko": "소고기", "en": "beef"},
        "input_aliases": {
            "ko": ["소고기", "쇠고기", "우육", "한우", "육우", "소갈비", "우갈비", "갈비살", "차돌", "차돌박이", "양지", "사태", "우삼겹", "부채살", "살치살", "안창살", "토시살", "치마살", "업진살", "제비추리", "안심", "등심", "꽃등심", "채끝", "채끝살", "새우살", "우설", "우족", "우꼬리", "대창", "곱창", "염통", "막창", "토시스테이크", "갈비탕", "육회", "샤브샤브", "스키야키", "스테이크", "불고기", "대창전골", "곱창전골"],
            "en": ["beef", "beef brisket", "beef rib", "short rib", "brisket", "striploin", "sirloin", "tenderloin", "ribeye", "chuck flap", "flank steak", "skirt steak", "hanger steak", "tri-tip", "beef tripe", "beef large intestine", "beef omasum", "beef tongue", "beef tail", "beef broth", "steak", "sukiyaki", "shabu shabu", "bulgogi", "yukhoe"],
        },
        "menu_evidence_aliases": {
            "ko": ["소고기", "쇠고기", "우육", "한우", "육우", "소갈비", "우갈비", "갈비살", "차돌", "차돌박이", "양지", "사태", "우삼겹", "부채살", "살치살", "안창살", "토시살", "치마살", "업진살", "제비추리", "안심", "등심", "꽃등심", "채끝", "채끝살", "새우살", "우설", "우족", "우꼬리", "소막창", "소대창", "소곱창", "소염통"],
            "en": ["beef", "beef brisket", "short rib", "brisket", "striploin", "sirloin", "tenderloin", "ribeye", "flank steak", "skirt steak", "hanger steak", "tri-tip", "beef tongue", "beef tail", "beef tripe", "beef large intestine"],
        },
        "menu_weak_aliases": {
            "ko": ["갈비탕", "불고기", "스테이크", "샤브샤브", "스키야키", "곱창", "대창", "막창", "염통"],
            "en": ["beef broth", "steak", "bulgogi", "shabu shabu", "sukiyaki", "tripe", "large intestine"],
        },
        "menu_prior_aliases": {"ko": ["갈비탕", "토시스테이크", "대창전골", "곱창전골", "곱창볶음", "육회", "스키야키", "샤브샤브", "스테이크", "불고기"], "en": ["yukhoe", "sukiyaki", "shabu shabu", "bulgogi"]},
    },
    "pork": {
        "display": {"ko": "돼지고기", "en": "pork"},
        "input_aliases": {
            "ko": ["돼지고기", "돈육", "돼지", "삼겹살", "삼겹", "오겹살", "목살", "항정살", "항정", "가브리살", "갈매기살", "앞다리살", "뒷다리살", "등갈비", "돼지갈비", "수육", "제육", "제육볶음", "두루치기", "돈까스", "돈카츠", "베이컨", "햄", "족발", "보쌈", "편육", "머리고기", "돼지껍데기", "껍데기", "꼬들살", "미추리", "오소리감투", "돼지막창", "돼지곱창", "막창", "곱창", "염통", "차슈", "돈코츠", "돼지국밥", "순대국"],
            "en": ["pork", "pig", "ham", "bacon", "pork belly", "pork loin", "pork shoulder", "jowl", "pork rib", "spare rib", "prosciutto", "lard", "char siu", "tonkotsu", "pork stock"],
        },
        "menu_evidence_aliases": {
            "ko": ["돼지고기", "돈육", "돼지", "삼겹살", "삼겹", "오겹살", "목살", "항정살", "항정", "가브리살", "갈매기살", "앞다리살", "뒷다리살", "돼지갈비", "제육", "제육볶음", "돈까스", "돈카츠", "족발", "보쌈", "편육", "머리고기", "돼지껍데기", "꼬들살", "미추리", "오소리감투", "돼지막창", "돼지곱창", "돼지염통", "차슈", "돈코츠"],
            "en": ["pork", "pig", "pork belly", "pork loin", "pork shoulder", "jowl", "pork rib", "spare rib", "prosciutto", "char siu", "tonkotsu"],
        },
        "menu_weak_aliases": {
            "ko": ["수육", "베이컨", "햄", "껍데기", "막창", "곱창", "염통"],
            "en": ["ham", "bacon", "lard", "pork stock"],
        },
        "menu_prior_aliases": {"ko": ["순대국", "돼지국밥", "보쌈정식", "제육볶음", "제육덮밥", "두루치기", "수육국밥", "차슈덮밥", "돈코츠라멘", "막창구이"], "en": ["char siu rice", "tonkotsu ramen"]},
    },
    "chicken": {
        "display": {"ko": "닭고기", "en": "chicken"},
        "input_aliases": {
            "ko": ["닭", "닭고기", "계육", "치킨", "닭다리", "닭다리살", "닭가슴살", "닭안심", "닭날개", "닭봉", "닭윙", "닭발", "닭목살", "닭근위", "닭염통", "모래집", "닭육수", "닭갈비", "닭볶음탕", "찜닭", "닭강정", "닭곰탕", "삼계탕", "백숙", "가라아게", "치킨가스", "치킨카츠", "치킨텐더", "치킨스테이크"],
            "en": ["chicken", "chicken thigh", "chicken breast", "chicken tender", "chicken wing", "drumstick", "gizzard", "chicken heart", "chicken stock", "poultry", "karaage", "fried chicken", "chicken katsu", "samgyetang"],
        },
        "menu_evidence_aliases": {
            "ko": ["닭", "닭고기", "계육", "치킨", "닭다리", "닭다리살", "닭가슴살", "닭안심", "닭날개", "닭봉", "닭윙", "닭발", "닭목살", "닭근위", "닭염통", "모래집"],
            "en": ["chicken", "chicken thigh", "chicken breast", "chicken tender", "chicken wing", "drumstick", "gizzard", "chicken heart", "fried chicken"],
        },
        "menu_weak_aliases": {
            "ko": ["닭육수"],
            "en": ["chicken stock", "poultry"],
        },
        "menu_prior_aliases": {"ko": ["삼계탕", "백숙", "찜닭", "닭갈비", "닭볶음탕", "닭강정", "닭곰탕", "가라아게", "치킨가스", "치킨카츠", "치킨텐더", "치킨스테이크"], "en": ["karaage", "samgyetang", "chicken katsu"]},
    },
    "lamb": {
        "display": {"ko": "양고기", "en": "lamb"},
        "input_aliases": {
            "ko": ["양고기", "양갈비", "램", "양육", "양꼬치", "양다리", "양사태", "양등심", "양어깨살", "프렌치랙", "램찹"],
            "en": ["lamb", "mutton", "lamb chop", "lamb rib", "lamb skewer", "leg of lamb", "lamb shank", "rack of lamb"],
        },
        "menu_evidence_aliases": {
            "ko": ["양고기", "양갈비", "램", "양육", "양꼬치", "양다리", "양사태", "양등심", "양어깨살", "프렌치랙", "램찹"],
            "en": ["lamb", "mutton", "lamb chop", "lamb rib", "lamb skewer", "leg of lamb", "lamb shank", "rack of lamb"],
        },
        "menu_weak_aliases": {"ko": [], "en": []},
        "menu_prior_aliases": {"ko": ["양꼬치", "램찹", "프렌치랙"], "en": ["lamb skewer", "rack of lamb"]},
    },
    "duck": {
        "display": {"ko": "오리고기", "en": "duck"},
        "input_aliases": {
            "ko": ["오리", "오리고기", "훈제오리", "오리훈제", "오리로스", "오리주물럭", "오리불고기", "오리백숙", "오리탕", "오리가슴살", "북경오리", "베이징덕"],
            "en": ["duck", "duck meat", "smoked duck", "duck breast", "roast duck", "peking duck"],
        },
        "menu_evidence_aliases": {
            "ko": ["오리", "오리고기", "훈제오리", "오리훈제", "오리로스", "오리주물럭", "오리불고기", "오리가슴살", "북경오리", "베이징덕"],
            "en": ["duck", "duck meat", "smoked duck", "duck breast", "roast duck", "peking duck"],
        },
        "menu_weak_aliases": {"ko": [], "en": []},
        "menu_prior_aliases": {"ko": ["오리탕", "오리백숙", "오리주물럭", "오리불고기", "북경오리", "베이징덕"], "en": ["peking duck"]},
    },
}


SPANISH_ALIAS_EXTENSIONS: Dict[str, Dict[str, object]] = {
    "egg": {
        "display": "huevo",
        "input_aliases": [
            "huevo",
            "huevos",
            "clara de huevo",
            "yema de huevo",
            "huevo frito",
            "huevo pochado",
            "huevo revuelto",
            "omelet",
            "tortilla",
            "tortilla española",
        ],
        "menu_evidence_aliases": [
            "huevo",
            "huevos",
            "clara de huevo",
            "yema de huevo",
        ],
        "menu_weak_aliases": [
            "huevo frito",
            "huevo pochado",
            "huevo revuelto",
            "omelet",
            "tortilla",
        ],
    },
    "dairy": {
        "display": "lácteos",
        "input_aliases": [
            "lacteo",
            "lácteo",
            "lacteos",
            "lácteos",
            "producto lacteo",
            "producto lácteo",
            "productos lacteos",
            "productos lácteos",
        ],
        "menu_evidence_aliases": [
            "lacteo",
            "lácteo",
            "lacteos",
            "lácteos",
        ],
        "menu_prior_aliases": [
            "alfredo",
            "salsa cremosa",
            "pasta cremosa",
            "lasaña",
            "pizza",
            "margherita",
            "malai",
            "gratinado",
        ],
    },
    "milk": {
        "display": "leche",
        "input_aliases": [
            "leche",
            "leche entera",
            "leche desnatada",
            "leche condensada",
            "leche evaporada",
            "lactosa",
            "nata",
            "crema de leche",
        ],
        "menu_evidence_aliases": [
            "leche",
            "leche condensada",
            "leche evaporada",
            "crema de leche",
        ],
        "menu_weak_aliases": [
            "nata",
            "crema",
        ],
        "menu_prior_aliases": [
            "té con leche",
            "cafe con leche",
            "café con leche",
            "batido de leche",
            "malai",
        ],
    },
    "cheese": {
        "display": "queso",
        "input_aliases": [
            "queso",
            "queso crema",
            "mozzarella",
            "parmesano",
            "cheddar",
            "manchego",
            "gouda",
            "queso rallado",
            "queso fundido",
        ],
        "menu_evidence_aliases": [
            "queso",
            "queso crema",
            "mozzarella",
            "parmesano",
            "cheddar",
            "manchego",
            "gouda",
        ],
        "menu_prior_aliases": [
            "pizza",
            "lasaña",
            "margherita",
            "tabla de quesos",
            "quesadilla",
        ],
    },
    "butter": {
        "display": "mantequilla",
        "input_aliases": [
            "mantequilla",
            "mantequilla salada",
            "mantequilla sin sal",
            "ghee",
        ],
        "menu_evidence_aliases": [
            "mantequilla",
            "mantequilla salada",
            "mantequilla sin sal",
            "ghee",
        ],
    },
    "peanut": {
        "display": "cacahuete",
        "input_aliases": [
            "cacahuete",
            "cacahuetes",
            "mani",
            "maní",
            "mantequilla de cacahuete",
            "mantequilla de maní",
        ],
        "menu_evidence_aliases": [
            "cacahuete",
            "cacahuetes",
            "mani",
            "maní",
            "mantequilla de cacahuete",
            "mantequilla de maní",
        ],
    },
    "tree nut": {
        "display": "frutos secos",
        "input_aliases": [
            "fruto seco",
            "frutos secos",
            "almendra",
            "almendras",
            "nuez",
            "nueces",
            "nuez pecana",
            "anacardo",
            "anacardos",
            "pistacho",
            "pistachos",
            "avellana",
            "avellanas",
            "piñon",
            "piñones",
            "macadamia",
        ],
        "menu_evidence_aliases": [
            "almendra",
            "almendras",
            "nuez",
            "nueces",
            "nuez pecana",
            "anacardo",
            "anacardos",
            "pistacho",
            "pistachos",
            "avellana",
            "avellanas",
            "piñon",
            "piñones",
            "macadamia",
        ],
    },
    "soy": {
        "display": "soja",
        "input_aliases": [
            "soja",
            "soya",
            "frijol de soja",
            "tofu",
            "tofu frito",
            "leche de soja",
            "miso",
            "salsa de soja",
            "edamame",
            "tempeh",
        ],
        "menu_evidence_aliases": [
            "soja",
            "soya",
            "tofu",
            "tofu frito",
            "leche de soja",
            "edamame",
            "tempeh",
        ],
        "menu_weak_aliases": [
            "miso",
            "salsa de soja",
        ],
        "menu_prior_aliases": [
            "sopa de miso",
        ],
    },
    "gluten": {
        "display": "gluten",
        "input_aliases": [
            "gluten",
            "glúten",
            "proteina de trigo",
            "proteína de trigo",
        ],
        "menu_evidence_aliases": [
            "gluten",
            "glúten",
            "proteina de trigo",
            "proteína de trigo",
        ],
    },
    "wheat": {
        "display": "trigo",
        "input_aliases": [
            "trigo",
            "harina",
            "harina de trigo",
            "gluten",
            "fideo",
            "fideos",
            "pasta",
            "udon",
            "ramen",
            "pan",
            "tostada",
            "rebozado",
            "empanado",
            "masa",
            "masa de empanada",
        ],
        "menu_evidence_aliases": [
            "trigo",
            "harina",
            "harina de trigo",
            "gluten",
        ],
        "menu_weak_aliases": [
            "fideo",
            "fideos",
            "pasta",
            "udon",
            "ramen",
            "pan",
            "tostada",
            "rebozado",
            "empanado",
            "masa",
            "masa de empanada",
        ],
    },
    "shrimp": {
        "display": "camarón",
        "input_aliases": [
            "camaron",
            "camarón",
            "camarones",
            "gamba",
            "gambas",
            "langostino",
            "langostinos",
            "camaron seco",
            "camarón seco",
            "pasta de camaron",
            "pasta de camarón",
        ],
        "menu_evidence_aliases": [
            "camaron",
            "camarón",
            "camarones",
            "gamba",
            "gambas",
            "langostino",
            "langostinos",
            "camaron seco",
            "camarón seco",
            "pasta de camaron",
            "pasta de camarón",
        ],
    },
    "crab": {
        "display": "cangrejo",
        "input_aliases": [
            "cangrejo",
            "carne de cangrejo",
            "cangrejo azul",
            "cangrejo real",
            "centollo",
            "jaiba",
        ],
        "menu_evidence_aliases": [
            "cangrejo",
            "carne de cangrejo",
            "cangrejo azul",
            "cangrejo real",
            "centollo",
            "jaiba",
        ],
    },
    "shellfish": {
        "display": "mariscos",
        "input_aliases": [
            "marisco",
            "mariscos",
            "molusco",
            "moluscos",
            "almeja",
            "almejas",
            "mejillon",
            "mejillón",
            "mejillones",
            "ostra",
            "ostras",
            "vieira",
            "vieiras",
            "berberecho",
            "berberechos",
            "navaja",
            "navajas",
            "abulon",
            "abulón",
        ],
        "menu_evidence_aliases": [
            "marisco",
            "mariscos",
            "molusco",
            "moluscos",
            "almeja",
            "almejas",
            "mejillon",
            "mejillón",
            "mejillones",
            "ostra",
            "ostras",
            "vieira",
            "vieiras",
            "berberecho",
            "berberechos",
            "navaja",
            "navajas",
            "abulon",
            "abulón",
        ],
    },
    "fish": {
        "display": "pescado",
        "input_aliases": [
            "pescado",
            "atun",
            "atún",
            "salmon",
            "salmón",
            "bacalao",
            "merluza",
            "dorada",
            "lubina",
            "anchoa",
            "anchoas",
            "sardina",
            "sardinas",
            "bonito",
            "pez espada",
            "rape",
        ],
        "menu_evidence_aliases": [
            "pescado",
            "atun",
            "atún",
            "salmon",
            "salmón",
            "bacalao",
            "merluza",
            "dorada",
            "lubina",
            "anchoa",
            "anchoas",
            "sardina",
            "sardinas",
            "bonito",
            "pez espada",
            "rape",
        ],
        "menu_weak_aliases": [
            "salsa de pescado",
            "caldo de pescado",
        ],
    },
    "sesame": {
        "display": "sésamo",
        "input_aliases": [
            "sesamo",
            "sésamo",
            "ajonjoli",
            "ajonjolí",
            "aceite de sesamo",
            "aceite de sésamo",
            "tahini",
        ],
        "menu_evidence_aliases": [
            "sesamo",
            "sésamo",
            "ajonjoli",
            "ajonjolí",
        ],
        "menu_weak_aliases": [
            "aceite de sesamo",
            "aceite de sésamo",
            "tahini",
        ],
    },
    "beef": {
        "display": "carne de res",
        "input_aliases": [
            "res",
            "carne de res",
            "ternera",
            "vacuno",
            "buey",
            "chuleta de res",
            "costilla de res",
            "brisket",
            "lomo de res",
            "solomillo",
            "entrecot",
            "entrecôte",
            "falda de res",
            "lengua de res",
            "rabo de toro",
            "callos",
            "churrasco",
            "bistec",
            "filete de res",
        ],
        "menu_evidence_aliases": [
            "res",
            "carne de res",
            "ternera",
            "vacuno",
            "buey",
            "chuleta de res",
            "costilla de res",
            "lomo de res",
            "solomillo",
            "entrecot",
            "entrecôte",
            "falda de res",
            "lengua de res",
            "rabo de toro",
            "churrasco",
            "bistec",
            "filete de res",
        ],
        "menu_weak_aliases": [
            "caldo de res",
            "callos",
            "steak",
            "filete",
        ],
        "menu_prior_aliases": [
            "carne asada",
            "estofado de ternera",
        ],
    },
    "pork": {
        "display": "cerdo",
        "input_aliases": [
            "cerdo",
            "carne de cerdo",
            "puerco",
            "chancho",
            "cochino",
            "jamon",
            "jamón",
            "jamon serrano",
            "jamón serrano",
            "serrano",
            "jamon iberico",
            "jamón ibérico",
            "iberico",
            "ibérico",
            "chorizo",
            "salchichon",
            "salchichón",
            "morcilla",
            "lomo de cerdo",
            "panceta",
            "tocino",
            "bacon",
            "jamon cocido",
            "jamón cocido",
            "jamon curado",
            "jamón curado",
            "prosciutto",
            "salami",
            "soppressata",
            "lardon",
            "lardón",
        ],
        "menu_evidence_aliases": [
            "cerdo",
            "carne de cerdo",
            "puerco",
            "chancho",
            "cochino",
            "jamon",
            "jamón",
            "jamon serrano",
            "jamón serrano",
            "serrano",
            "jamon iberico",
            "jamón ibérico",
            "iberico",
            "ibérico",
            "chorizo",
            "salchichon",
            "salchichón",
            "morcilla",
            "lomo de cerdo",
            "panceta",
            "tocino",
            "bacon",
            "prosciutto",
            "salami",
            "soppressata",
        ],
        "menu_weak_aliases": [
            "jamon cocido",
            "jamón cocido",
            "jamon curado",
            "jamón curado",
            "grasa de cerdo",
            "caldo de cerdo",
        ],
        "menu_prior_aliases": [
            "caldo gallego",
            "tabla iberica",
            "tabla de serrano",
            "croquetas de jamon",
            "croquetas de jamón",
        ],
    },
    "chicken": {
        "display": "pollo",
        "input_aliases": [
            "pollo",
            "carne de pollo",
            "muslo de pollo",
            "pechuga de pollo",
            "alita de pollo",
            "pollo frito",
            "pollo asado",
            "caldo de pollo",
            "gallina",
            "ave",
        ],
        "menu_evidence_aliases": [
            "pollo",
            "carne de pollo",
            "muslo de pollo",
            "pechuga de pollo",
            "alita de pollo",
            "pollo frito",
            "pollo asado",
            "gallina",
        ],
        "menu_weak_aliases": [
            "caldo de pollo",
            "ave",
        ],
        "menu_prior_aliases": [
            "katsu de pollo",
            "karaage de pollo",
        ],
    },
    "lamb": {
        "display": "cordero",
        "input_aliases": [
            "cordero",
            "carne de cordero",
            "lechazo",
            "borrego",
            "chuleta de cordero",
            "costilla de cordero",
            "pierna de cordero",
            "paletilla de cordero",
        ],
        "menu_evidence_aliases": [
            "cordero",
            "carne de cordero",
            "lechazo",
            "borrego",
            "chuleta de cordero",
            "costilla de cordero",
            "pierna de cordero",
            "paletilla de cordero",
        ],
        "menu_prior_aliases": [
            "asado de cordero",
            "chuletillas de cordero",
        ],
    },
    "duck": {
        "display": "pato",
        "input_aliases": [
            "pato",
            "carne de pato",
            "pato ahumado",
            "pechuga de pato",
            "pato asado",
            "pato pekin",
            "pato pequines",
            "pato pequinés",
        ],
        "menu_evidence_aliases": [
            "pato",
            "carne de pato",
            "pato ahumado",
            "pechuga de pato",
            "pato asado",
            "pato pekin",
            "pato pequines",
            "pato pequinés",
        ],
        "menu_prior_aliases": [
            "pato laqueado",
            "pato pekin",
        ],
    },
}


def _dedupe_localized_terms(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if not isinstance(value, str):
            continue
        cleaned = re.sub(r"\s+", " ", value).strip()
        key = normalize_ingredient_token(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


def _merge_spanish_terms(meta: Dict[str, object], section: str, additions: Iterable[str]) -> None:
    section_values = meta.get(section, {})
    if not isinstance(section_values, dict):
        section_values = {}
        meta[section] = section_values

    existing = section_values.get("es", [])
    current_terms: List[str] = []
    if isinstance(existing, str):
        current_terms = [existing]
    elif isinstance(existing, list):
        current_terms = [term for term in existing if isinstance(term, str)]
    current_terms.extend(additions)
    section_values["es"] = _dedupe_localized_terms(current_terms)


def _apply_spanish_alias_extensions(catalog: Dict[str, Dict[str, object]]) -> None:
    for canonical, extension in SPANISH_ALIAS_EXTENSIONS.items():
        meta = catalog.get(canonical)
        if not isinstance(meta, dict):
            continue

        display_value = extension.get("display")
        if isinstance(display_value, str) and display_value.strip():
            display = meta.get("display", {})
            if not isinstance(display, dict):
                display = {}
                meta["display"] = display
            display["es"] = display_value.strip()

        for section in ("input_aliases", "menu_evidence_aliases", "menu_weak_aliases", "menu_prior_aliases"):
            terms = extension.get(section)
            if isinstance(terms, list):
                _merge_spanish_terms(meta, section, terms)


_apply_spanish_alias_extensions(AVOID_INGREDIENT_CATALOG)


def _iter_alias_terms(
    section: Literal["display", "input_aliases", "menu_evidence_aliases", "menu_weak_aliases"],
) -> Iterable[tuple[str, str]]:
    for canonical, meta in AVOID_INGREDIENT_CATALOG.items():
        values = meta.get(section, {})
        for term in _iter_supported_localized_terms(values):
            yield canonical, term


def _build_lookup(
    sections: List[Literal["display", "input_aliases", "menu_evidence_aliases", "menu_weak_aliases"]],
) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for section in sections:
        for canonical, term in _iter_alias_terms(section):
            normalized = normalize_ingredient_token(term)
            if normalized:
                lookup[normalized] = canonical

    for canonical in AVOID_INGREDIENT_CATALOG.keys():
        normalized_canonical = normalize_ingredient_token(canonical)
        if normalized_canonical:
            lookup[normalized_canonical] = canonical
    return lookup


INPUT_ALIAS_LOOKUP = _build_lookup(["display", "input_aliases"])
MENU_EVIDENCE_LOOKUP = _build_lookup(["display", "menu_evidence_aliases"])
MENU_ALL_ALIAS_LOOKUP = _build_lookup(["display", "menu_evidence_aliases", "menu_weak_aliases"])


def build_avoid_synonym_lookup() -> Dict[str, str]:
    # 하위 호환: 기존 함수명은 사용자 입력 정규화용 lookup을 반환한다.
    return dict(INPUT_ALIAS_LOOKUP)


def build_menu_evidence_lookup(include_weak: bool = False) -> Dict[str, str]:
    return dict(MENU_ALL_ALIAS_LOOKUP if include_weak else MENU_EVIDENCE_LOOKUP)


def get_menu_evidence_catalog() -> Dict[str, Dict[str, List[str]]]:
    catalog: Dict[str, Dict[str, List[str]]] = {}
    for canonical, meta in AVOID_INGREDIENT_CATALOG.items():
        display = meta.get("display", {})
        if not isinstance(display, dict):
            display = {}

        strong_terms: List[str] = []
        prior_terms: List[str] = []
        weak_terms: List[str] = []
        direct_terms: List[str] = []

        for value in _iter_supported_localized_terms(display):
            direct_terms.append(value.strip())

        direct_terms.append(canonical)

        for section_name, bucket in (
            ("menu_evidence_aliases", strong_terms),
            ("menu_prior_aliases", prior_terms),
            ("menu_weak_aliases", weak_terms),
        ):
            values = meta.get(section_name, {})
            for term in _iter_supported_localized_terms(values):
                bucket.append(term)

        def _dedupe(values: List[str]) -> List[str]:
            out: List[str] = []
            seen = set()
            for value in values:
                key = normalize_ingredient_token(value)
                if not key or key in seen:
                    continue
                seen.add(key)
                out.append(value)
            return out

        catalog[canonical] = {
            "direct": _dedupe(direct_terms),
            "strong": _dedupe(strong_terms),
            "prior": _dedupe(prior_terms),
            "weak": _dedupe(weak_terms),
        }
    return catalog


def get_canonical_ingredient(text: str, mode: AliasMode = "input") -> Optional[str]:
    normalized = normalize_ingredient_token(text)
    if not normalized:
        return None

    if mode == "input":
        return INPUT_ALIAS_LOOKUP.get(normalized)
    if mode == "menu_strong":
        return MENU_EVIDENCE_LOOKUP.get(normalized)
    return MENU_ALL_ALIAS_LOOKUP.get(normalized)


def get_canonical_parents(canonical: str) -> List[str]:
    meta = AVOID_INGREDIENT_CATALOG.get(canonical, {})
    parents = meta.get("parents", [])
    if not isinstance(parents, list):
        return []
    out: List[str] = []
    for parent in parents:
        if isinstance(parent, str) and parent.strip():
            out.append(parent.strip().casefold())
    return out


def get_canonical_ancestors(canonical: str) -> List[str]:
    canonical_norm = (canonical or "").strip().casefold()
    if not canonical_norm:
        return []

    ordered: List[str] = []
    seen = set()
    stack = list(get_canonical_parents(canonical_norm))
    while stack:
        current = stack.pop(0)
        if current in seen:
            continue
        seen.add(current)
        ordered.append(current)
        stack.extend(get_canonical_parents(current))
    return ordered


def _canonical_distance_map(canonical: str) -> Dict[str, int]:
    canonical_norm = (canonical or "").strip().casefold()
    if not canonical_norm:
        return {}

    distance_map: Dict[str, int] = {canonical_norm: 0}
    queue: List[str] = [canonical_norm]
    while queue:
        current = queue.pop(0)
        current_dist = distance_map[current]
        for parent in get_canonical_parents(current):
            if parent in distance_map:
                continue
            distance_map[parent] = current_dist + 1
            queue.append(parent)
    return distance_map


def find_matching_avoid_canonical(canonical: str, allowed_canonicals: Iterable[str]) -> Optional[str]:
    allowed = {
        str(value).strip().casefold()
        for value in allowed_canonicals
        if isinstance(value, str) and value.strip()
    }
    canonical_norm = (canonical or "").strip().casefold()
    if not canonical_norm:
        return None

    # 0) exact
    if canonical_norm in allowed:
        return canonical_norm

    source_dist = _canonical_distance_map(canonical_norm)
    if not source_dist:
        return None

    best: Optional[tuple[int, int, str]] = None
    for allowed_canonical in sorted(allowed):
        target_dist = _canonical_distance_map(allowed_canonical)
        if not target_dist:
            continue

        # 1) inferred child -> allowed ancestor (e.g., cheese -> dairy)
        if allowed_canonical in source_dist:
            candidate = (1, source_dist[allowed_canonical], allowed_canonical)
        # 2) inferred ancestor -> allowed child (e.g., dairy -> milk)
        elif canonical_norm in target_dist:
            candidate = (2, target_dist[canonical_norm], allowed_canonical)
        else:
            # 3) same family sibling match via shared ancestor (e.g., cheese <-> milk via dairy)
            shared = set(source_dist.keys()) & set(target_dist.keys())
            shared.discard(canonical_norm)
            shared.discard(allowed_canonical)
            if not shared:
                continue
            family_distance = min(source_dist[key] + target_dist[key] for key in shared)
            candidate = (3, family_distance, allowed_canonical)

        if best is None or candidate < best:
            best = candidate

    return best[2] if best is not None else None


def get_display_name(canonical: str, lang: str = "en") -> str:
    meta = AVOID_INGREDIENT_CATALOG.get(canonical, {})
    display = meta.get("display", {})
    if not isinstance(display, dict):
        return canonical

    if lang in display and isinstance(display[lang], str) and display[lang].strip():
        return display[lang].strip()

    for fallback in ("en", "ko", "es"):
        value = display.get(fallback)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return canonical


def canonicalize_avoid_ingredients(ingredients: Iterable[str]) -> List[str]:
    canonicalized: List[str] = []
    seen = set()

    for ingredient in ingredients:
        normalized = normalize_ingredient_token(ingredient)
        if not normalized:
            continue
        canonical = INPUT_ALIAS_LOOKUP.get(normalized, re.sub(r"\s+", " ", str(ingredient)).strip())
        if canonical and canonical not in seen:
            seen.add(canonical)
            canonicalized.append(canonical)

    return canonicalized


def build_canonical_display_map(ingredients: Iterable[str], lang: str = "en") -> Dict[str, str]:
    display_map: Dict[str, str] = {}

    for ingredient in ingredients:
        normalized = normalize_ingredient_token(ingredient)
        if not normalized:
            continue

        raw_display = re.sub(r"\s+", " ", str(ingredient)).strip()
        canonical = INPUT_ALIAS_LOOKUP.get(normalized, raw_display).casefold()
        if canonical and canonical not in display_map:
            display_map[canonical] = raw_display

    for canonical in canonicalize_avoid_ingredients(ingredients):
        if canonical not in display_map:
            display_map[canonical] = get_display_name(canonical, lang=lang)

    return display_map


def build_avoid_lookup(ingredients: Iterable[str]) -> Dict[str, str]:
    avoid_lookup: Dict[str, str] = {}

    for ingredient in ingredients:
        normalized_original = normalize_ingredient_token(ingredient)
        if not normalized_original:
            continue

        display_name = re.sub(r"\s+", " ", str(ingredient)).strip()
        canonical = INPUT_ALIAS_LOOKUP.get(normalized_original, display_name)
        for token, mapped in INPUT_ALIAS_LOOKUP.items():
            if mapped == canonical:
                avoid_lookup[token] = display_name

        normalized_canonical = normalize_ingredient_token(canonical)
        if normalized_canonical:
            avoid_lookup[normalized_canonical] = display_name

        avoid_lookup[normalized_original] = display_name

    return avoid_lookup


def get_catalog_stats() -> Dict[str, int]:
    return {
        "canonical_count": len(AVOID_INGREDIENT_CATALOG),
        "input_alias_count": len(INPUT_ALIAS_LOOKUP),
        "menu_evidence_alias_count": len(MENU_EVIDENCE_LOOKUP),
        "menu_all_alias_count": len(MENU_ALL_ALIAS_LOOKUP),
    }


# 하위 호환: 예전 flat dict를 참조하는 코드는 input_aliases 기준으로만 보게 한다.
AVOID_INGREDIENT_SYNONYMS: Dict[str, List[str]] = {}
for _canonical, _meta in AVOID_INGREDIENT_CATALOG.items():
    _synonyms: List[str] = []
    _display = _meta.get("display", {})
    if isinstance(_display, dict):
        for _value in _display.values():
            if isinstance(_value, str):
                _synonyms.append(_value)
    _localized_synonyms = _meta.get("input_aliases", {})
    if isinstance(_localized_synonyms, dict):
        for _values in _localized_synonyms.values():
            if isinstance(_values, list):
                _synonyms.extend([_value for _value in _values if isinstance(_value, str)])
    deduped: List[str] = []
    seen = set()
    for _value in _synonyms:
        _key = normalize_ingredient_token(_value)
        if not _key or _key in seen:
            continue
        seen.add(_key)
        deduped.append(_value)
    AVOID_INGREDIENT_SYNONYMS[_canonical] = deduped
