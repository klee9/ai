import argparse
import json
import os
import sys
from pathlib import Path

from app.agents._0_orchestrator import MenuAgentOrchestrator
from app.clients.gemma_client import GemmaClient
from app.utils.env_loader import load_local_env


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
        choices=["ko", "en", "es"],
        help="사용자/응답 언어",
    )
    parser.add_argument(
        "--menu-country-code",
        default="AUTO",
        help="메뉴판 OCR 언어 힌트. 모르면 AUTO",
    )
    parser.add_argument(
        "--menu-lang",
        default="auto",
        choices=["auto", "ko", "en", "es"],
        help="메뉴판 OCR 언어. 주면 자동 언어탐지를 생략",
    )
    parser.add_argument(
        "--avoid",
        nargs="*",
        default=["계란", "우유"],
        help="기피 재료 목록. 예: --avoid 계란 우유 땅콩",
    )
    parser.add_argument(
        "--presigned-url",
        default="",
        help="bbox 결과를 업로드할 presigned PUT URL(옵션)",
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
    parser.add_argument(
        "--debug-translation",
        action="store_true",
        help="번역 단계 입력/출력/provider와 최종 menu 값을 stderr로 출력",
    )
    return parser


def require_api_key():
    api_key = os.getenv("GOOGLE_API_KEY")
    if api_key:
        return api_key
    raise SystemExit(
        "[ERROR] GOOGLE_API_KEY가 설정되어 있지 않습니다.\n"
        "예: export GOOGLE_API_KEY='your-key'\n"
        "또는 프로젝트 루트 .env에 GOOGLE_API_KEY=... 를 추가하세요."
    )


def print_summary(result, args, resolved_image_source: str):
    print("=" * 72)
    print("Full Cycle Test")
    print("=" * 72)
    print(f"Image Source      : {resolved_image_source}")
    print(f"User Language     : {args.user_lang}")
    print(f"Menu Lang Hint    : {args.menu_lang}")
    print(f"Output Language   : {result.output_lang or '-'}")
    print(f"Menu Country Code : {result.menu_country_code or '-'}")
    print(f"Menu OCR Lang     : {result.menu_ocr_lang or '-'} ({result.menu_ocr_lang_source or '-'})")
    print(f"Avoid Ingredients : {', '.join(args.avoid) if args.avoid else '-'}")
    print(f"OCR Timeout(sec)  : {os.getenv('OCR_VISION_TIMEOUT_SEC', '-')}")
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

    print("[Top3 BBox]")
    print(f"url={result.bbox_image_url or '-'}")
    print(f"local_path={result.bbox_image_local_path or '-'}")
    print(f"targets={result.bbox_target_menus or []}")
    print()

    print("[Timings ms]")
    for key, value in (result.timings_ms or {}).items():
        print(f"- {key}: {value}")


def main():
    loaded_env_path = load_local_env()
    parser = build_parser()
    args = parser.parse_args()

    image_source = args.image
    if not (image_source.startswith("http://") or image_source.startswith("https://")):
        image_source = str(Path(image_source).expanduser().resolve())

    api_key = require_api_key()
    if loaded_env_path:
        print(f"[INFO] Loaded env file: {loaded_env_path}")
    if os.getenv("GOOGLE_VISION_API_KEY"):
        print("[INFO] Vision key source: GOOGLE_VISION_API_KEY")
    else:
        print("[INFO] Vision key source: GOOGLE_API_KEY fallback")
    gemma = GemmaClient(api_key=api_key, model=args.model)
    orchestrator = MenuAgentOrchestrator(gemma, uncertainty_penalty=40)
    if args.debug_translation:
        _install_translation_debug_hooks(orchestrator)

    result = orchestrator.run(
        image_url=image_source,
        avoid=args.avoid,
        user_lang=args.user_lang,
        menu_lang=args.menu_lang,
        menu_country_code=args.menu_country_code,
        presigned_url=args.presigned_url,
    )

    if args.debug_translation:
        _print_final_menu_debug(result)

    if args.json:
        payload = result.model_dump() if hasattr(result, "model_dump") else result.dict()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print_summary(result, args, image_source)

def _install_translation_debug_hooks(orchestrator: MenuAgentOrchestrator) -> None:
    translate_agent = orchestrator.translate_agent
    original_google = translate_agent._translate_with_google
    original_run = translate_agent.run

    def debug_google(*args, **kwargs):
        texts = kwargs.get("texts") or (args[0] if args else [])
        source_lang = kwargs.get("source_lang") or (args[1] if len(args) > 1 else "auto")
        target_lang = kwargs.get("target_lang") or (args[2] if len(args) > 2 else "en")
        print(
            f"[DEBUG][GOOGLE] request count={len(texts)} source={source_lang} target={target_lang}",
            file=sys.stderr,
        )
        try:
            result = original_google(*args, **kwargs)
        except Exception as exc:
            print(f"[DEBUG][GOOGLE] error={type(exc).__name__}: {exc}", file=sys.stderr)
            raise
        print(f"[DEBUG][GOOGLE] success count={len(result)}", file=sys.stderr)
        for src, item in result.items():
            print(
                f"[DEBUG][GOOGLE] {src} => {item.translated} | provider={item.provider}",
                file=sys.stderr,
            )
        return result

    def debug_run(request):
        print(
            f"[DEBUG][TRANSLATE] source={request.source_lang} target={request.target_lang} count={len(request.texts)}",
            file=sys.stderr,
        )
        for src in request.texts:
            print(f"[DEBUG][TRANSLATE][IN] {src}", file=sys.stderr)
        output = original_run(request)
        for item in output.items:
            print(
                f"[DEBUG][TRANSLATE][OUT] {item.original} => {item.translated} | provider={item.provider}",
                file=sys.stderr,
            )
        return output

    translate_agent._translate_with_google = debug_google
    translate_agent.run = debug_run


def _print_final_menu_debug(result) -> None:
    print("[DEBUG][FINAL] ranked menus", file=sys.stderr)
    for item in result.items:
        print(
            f"[DEBUG][FINAL] menu={item.menu} | original={item.menu_original} | "
            f"score={item.score} | risk={item.risk}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
