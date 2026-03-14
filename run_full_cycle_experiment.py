import argparse
import json
import os
from pathlib import Path

from app.agents._0_orchestrator import MenuAgentOrchestrator
from app.clients.gemma_client import GemmaClient


def build_parser():
    parser = argparse.ArgumentParser(
        description="메뉴 이미지로 전체 추천 사이클을 로컬에서 테스트합니다."
    )
    parser.add_argument(
        "--image",
        default="menu_image/menu0.jpeg",
        help="입력 이미지 경로 또는 URL",
    )
    parser.add_argument(
        "--user-lang",
        default="ko",
        choices=["ko", "en", "cn"],
        help="사용자/응답 언어",
    )
    parser.add_argument(
        "--menu-country-code",
        default="KR",
        help="메뉴판 언어 국가 코드(예: KR, US, JP)",
    )
    parser.add_argument(
        "--avoid",
        nargs="*",
        default=["계란", "우유"],
        help="기피 재료 목록. 예: --avoid 계란 우유 땅콩",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_ID", "gemma-3-4b-it"),
        help="Gemma 모델 ID",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="최종 결과를 JSON으로만 출력",
    )
    return parser


def require_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    raise SystemExit(
        "[ERROR] GOOGLE_API_KEY가 설정되어 있지 않습니다.\n"
        "예: export GOOGLE_API_KEY='your-key'"
    )


def print_summary(result, args, resolved_image_source: str):
    print("=" * 72)
    print("Full Cycle Test")
    print("=" * 72)
    print(f"Image Source      : {resolved_image_source}")
    print(f"User Language     : {args.user_lang}")
    print(f"Menu Country Code : {args.menu_country_code}")
    print(f"Avoid Ingredients : {', '.join(args.avoid) if args.avoid else '-'}")
    print()

    print("[Extracted Menus]")
    extracted = result.items_extracted or []
    if not extracted:
        print("- 없음")
    else:
        for idx, menu in enumerate(extracted, start=1):
            print(f"{idx:>2}. {menu}")
    print()

    print("[Ranked Items]")
    if not result.items:
        print("- 결과 없음")
    else:
        for idx, item in enumerate(result.items, start=1):
            print(
                f"{idx:>2}. menu={item.menu} | original={item.menu_original} | "
                f"score={item.score} | risk={item.risk} | conf={item.confidence:.2f}"
            )
            print(
                f"    matched_avoid={item.matched_avoid} | "
                f"suspected_ingredients={item.suspected_ingredients}"
            )
            print(f"    reason={item.reason}")
    print()

    print("[Best]")
    if result.best is None:
        print("- 없음")
    else:
        print(
            f"menu={result.best.menu} | original={result.best.menu_original} | "
            f"score={result.best.score} | risk={result.best.risk} | "
            f"conf={result.best.confidence:.2f}"
        )
        print(f"reason={result.best.reason}")
    print()

    print("[Timings ms]")
    for key, value in (result.timings_ms or {}).items():
        print(f"- {key}: {value}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    image_source = args.image
    if not (image_source.startswith("http://") or image_source.startswith("https://")):
        image_source = str(Path(image_source).expanduser().resolve())

    api_key = require_api_key()
    gemma = GemmaClient(api_key=api_key, model=args.model)
    orchestrator = MenuAgentOrchestrator(gemma, uncertainty_penalty=40)

    result = orchestrator.run(
        image_url=image_source,
        avoid=args.avoid,
        user_lang=args.user_lang,
        menu_country_code=args.menu_country_code,
    )

    if args.json:
        payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print_summary(result, args, image_source)


if __name__ == "__main__":
    main()
