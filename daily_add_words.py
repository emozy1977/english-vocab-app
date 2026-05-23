from __future__ import annotations

import argparse
import sys

import pandas as pd

from app import config, generate_ai_words, load_words, today


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AIで英単語を毎日追加します。")
    parser.add_argument("--count", type=int, default=5, help="追加する単語数。初期値は5。")
    parser.add_argument("--category", default="", help="カテゴリの希望。例: Business, Academic")
    parser.add_argument("--difficulty", default="3から5を中心にする", help="難易度の希望。")
    parser.add_argument("--model", default=config("OPENAI_MODEL", "gpt-5.4-mini"), help="OpenAI APIのモデル名。")
    parser.add_argument("--force", action="store_true", help="今日すでに追加済みでも実行する。")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        df = load_words()
        df, added = generate_ai_words(df, args.count, args.category, args.difficulty, args.model)
    except Exception as exc:
        print(f"AI単語追加に失敗しました: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"{len(added)}語を追加しました: {', '.join(added)}" if added else f"{today()} 新しい単語は追加されませんでした。")


if __name__ == "__main__":
    main()
