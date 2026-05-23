from __future__ import annotations

import pandas as pd

from app import COLUMNS, DATA_FILE, normalize_df, save_words


def main() -> None:
    if not DATA_FILE.exists():
        raise SystemExit(f"{DATA_FILE} が見つかりません。")

    df = pd.read_csv(DATA_FILE, keep_default_na=False)
    df = normalize_df(df)
    save_words(df[COLUMNS])
    print(f"{len(df)}語をSupabaseへ保存しました。")


if __name__ == "__main__":
    main()
