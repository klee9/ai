import argparse
import json
import os

from app.agents._chat_1_avoid_taker import AvoidIntakeAgent
from app.agents._0_contracts import AvoidIntakeInput
from app.clients.gemma_client import GemmaClient


def build_parser():
    parser = argparse.ArgumentParser(description="avoid_intake_agent 로컬 질의 실험")
    parser.add_argument("--text", default="", help="단일 테스트 입력 문장")
    parser.add_argument("--lang", default="ko", help="응답 언어(ko/en/cn)")
    return parser


def to_dict(model):
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def run_once(agent: AvoidIntakeAgent, text: str, lang: str):
    req = AvoidIntakeInput(user_text=text, lang=lang)
    out = agent.run(req)
    data = to_dict(out)
    print(json.dumps(data, ensure_ascii=False, indent=2))


def run_repl(agent: AvoidIntakeAgent, lang: str):
    print("=== avoid_intake_agent REPL ===")
    print("종료: /q 또는 /quit 입력")
    while True:
        try:
            user_text = input("\n질문 입력 > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n종료합니다.")
            return

        if user_text in {"/q", "/quit"}:
            print("종료합니다.")
            return

        if not user_text:
            print("빈 입력입니다. 다시 입력해 주세요.")
            continue

        run_once(agent, user_text, lang)


def main():
    args = build_parser().parse_args()

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY가 설정되어 있지 않습니다.")
        print("예: export GOOGLE_API_KEY='your-key'")
        return 1

    lang = args.lang if args.lang in {"ko", "en", "cn"} else "ko"
    model_id = os.getenv("MODEL_ID", "gemma-3-4b-it")

    gemma = GemmaClient(api_key=api_key, model=model_id)
    agent = AvoidIntakeAgent(gemma)

    if args.text.strip():
        run_once(agent, args.text.strip(), lang)
        return 0

    run_repl(agent, lang)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
