# 英単語帳アプリ

スマホのブラウザで使える、Streamlit製の英単語帳アプリです。ローカルでは `words.csv`、クラウドではSupabaseに保存できます。

## 機能

- 単語登録
- 学習カード
- 筆記問題
- 穴埋め問題
- 苦手単語を優先しつつ他の単語も混ぜる復習
- 正解数、不正解数、最終学習日の保存
- OpenAI APIによるAI単語追加
- Supabase保存によるクラウド公開対応
- 任意のアプリパスワード保護

## ファイル構成

```text
.
├── app.py
├── daily_add_words.py
├── migrate_csv_to_supabase.py
├── supabase_schema.sql
├── words.csv
├── requirements.txt
└── README.md
```

## Macでの起動方法

```bash
cd /Users/emotoshizuo/Documents/Codex
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

ブラウザで開きます。

```text
http://localhost:8501
```

同じWi-Fiのスマホから見る場合は次で起動します。

```bash
streamlit run app.py --server.address 0.0.0.0
```

## Supabaseで外出先スマホ対応にする方法

外出先でスマホだけで使うには、アプリをStreamlit Community Cloudなどに公開し、単語データをSupabaseに保存します。

### 1. Supabaseプロジェクトを作成

[Supabase](https://supabase.com/) で新しいプロジェクトを作成します。

作成後、Supabaseのダッシュボードで `SQL Editor` を開き、`supabase_schema.sql` の内容を実行します。

### 2. Supabaseの接続情報を確認

Supabaseの `Project Settings` → `Data API` または `API` で次を確認します。

- Project URL
- `service_role` key

`service_role` key は強い権限を持つ秘密情報です。GitHubや公開ファイルには絶対に書かないでください。

### 3. ローカルでSupabase接続を試す

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
export APP_PASSWORD="好きなパスワード"
python migrate_csv_to_supabase.py
streamlit run app.py
```

画面上部に `保存先: Supabase` と表示されれば成功です。

## Streamlit Community Cloudに公開

1. [Streamlit Community Cloud](https://streamlit.io/cloud) にログインします。
2. `New app` を押します。
3. Repository に `emozy1977/english-vocab-app` を選びます。
4. Main file path に `app.py` を指定します。
5. Advanced settings の Secrets に次を入力します。

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
APP_PASSWORD = "好きなパスワード"
```

AI単語追加も使う場合は追加します。

```toml
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.4-mini"
```

`SUPABASE_SERVICE_ROLE_KEY` と `OPENAI_API_KEY` はSecretsだけに入れてください。GitHubには置きません。

デプロイ後に発行されるURLをスマホで開きます。

```text
https://your-app-name.streamlit.app
```

## AIで単語を追加する方法

`OPENAI_API_KEY` を設定した状態でアプリを起動すると、画面上部の `AI追加` から単語を増やせます。

ターミナルから手動で5語追加する場合は次を実行します。

```bash
python daily_add_words.py
```

## データ保存

`SUPABASE_URL` と `SUPABASE_SERVICE_ROLE_KEY` が設定されている場合はSupabaseの `words` テーブルに保存します。設定されていない場合は `words.csv` に保存します。

主な列は次の通りです。

- `word`: 英単語
- `pronunciation`: 発音メモ
- `meaning_ja`: 日本語の意味
- `example_en`: 英語の例文
- `example_ja`: 例文の日本語訳
- `category`: カテゴリ
- `difficulty`: 難易度
- `correct_count`: 正解数
- `wrong_count`: 不正解数
- `last_studied`: 最終学習日
