# 英単語帳アプリ

スマホのブラウザで使える、Streamlit製の英単語帳アプリです。単語データと学習履歴は `words.csv` に保存されます。

## 機能

- 単語登録
- 学習カード
- 筆記問題
- 穴埋め問題
- 聞き取り問題
- `wrong_count` が多い単語を優先する復習
- 正解数、不正解数、最終学習日のCSV保存
- OpenAI APIによるAI単語追加
- AIによる既存単語の穴埋め例文作り直し
- Macでの毎日5語自動追加
- Supabase保存によるクラウド公開対応
- 任意のアプリパスワード保護

## ファイル構成

```text
.
├── app.py
├── daily_add_words.py
├── migrate_csv_to_supabase.py
├── scripts/
│   └── auto_improve.py
├── tests/
│   └── test_smoke.py
├── docs/
│   └── auto-improvement.md
├── IMPROVEMENT_BACKLOG.md
├── .github/workflows/
│   └── daily-auto-improvement.yml
├── supabase_schema.sql
├── words.csv
├── requirements.txt
└── README.md
```

## Macでの起動方法

ターミナルでこのフォルダに移動します。

```bash
cd /path/to/english-vocab-app
```

仮想環境を作成して有効化します。

```bash
python3 -m venv .venv
source .venv/bin/activate
```

必要なライブラリをインストールします。

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

AI単語追加を使う場合は、OpenAI APIキーを環境変数に設定します。

```bash
export OPENAI_API_KEY="sk-..."
```

アプリを起動します。

```bash
streamlit run app.py
```

ブラウザで次のURLを開きます。

```text
http://localhost:8501
```

## テスト

最低限の動作確認として、アプリの主要な補助関数を読み込むsmoke testを用意しています。

```bash
python -B -m unittest discover -s tests
```

## 毎日自動改善PR

GitHub Actionsで、毎日1回だけ小さな改善PRを作る仕組みを用意しています。OpenAI APIを使って `IMPROVEMENT_BACKLOG.md` から安全な小タスクを1つ選び、テストが通った場合だけPRを作成します。

詳しい使い方、必要なGitHub Secrets、止め方は [docs/auto-improvement.md](docs/auto-improvement.md) を見てください。

## スマホから同じMac上のアプリを見る方法

Macとスマホを同じWi-Fiに接続します。MacのIPアドレスを確認します。

```bash
ipconfig getifaddr en0
```

次のように起動します。

```bash
streamlit run app.py --server.address 0.0.0.0
```

スマホのブラウザで次のURLを開きます。`<MacのIPアドレス>` は上で確認した値に置き換えてください。

```text
http://<MacのIPアドレス>:8501
```

## AIで単語を追加する方法

`OPENAI_API_KEY` を設定した状態でアプリを起動すると、画面上部の `AI追加` から単語を増やせます。

`AI追加` の `穴埋め例文を作り直す` から、既存単語すべての穴埋め例文をビジネス/IT向けの自然な文章に作り直せます。古い不自然なテンプレート例文が残っている場合は、このボタンを実行してください。

聞き取り問題では、穴埋め例文の英文を音声で再生し、聞こえた英文全体を入力して練習できます。判定では大文字小文字、句読点、余分な空白は無視します。

デフォルトでは `gpt-5.4-mini` を使います。別のモデルを使う場合は、次のように設定できます。

```bash
export OPENAI_MODEL="gpt-5.4-mini"
```

ターミナルから手動で5語追加する場合は次を実行します。

```bash
python daily_add_words.py
```

今日すでに追加済みでも追加したい場合は `--force` を付けます。

```bash
python daily_add_words.py --force
```

## Macで毎日5語を自動追加する方法

Mac標準の `launchd` を使うと、毎日自動で `daily_add_words.py` を実行できます。

まず、APIキーを保存するための環境変数ファイルを作ります。

```bash
mkdir -p ~/.english_vocab_app
cat > ~/.english_vocab_app/env.sh <<'EOF'
export OPENAI_API_KEY="sk-..."
export OPENAI_MODEL="gpt-5.4-mini"
EOF
chmod 600 ~/.english_vocab_app/env.sh
```

次に、LaunchAgentを作成します。以下の例では毎朝7時に5語追加します。

```bash
cat > ~/Library/LaunchAgents/com.english-vocab.daily-ai-words.plist <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>com.english-vocab.daily-ai-words</string>

  <key>ProgramArguments</key>
  <array>
    <string>/bin/zsh</string>
    <string>-lc</string>
    <string>source ~/.english_vocab_app/env.sh && cd /path/to/english-vocab-app && .venv/bin/python daily_add_words.py</string>
  </array>

  <key>StartCalendarInterval</key>
  <dict>
    <key>Hour</key>
    <integer>7</integer>
    <key>Minute</key>
    <integer>0</integer>
  </dict>

  <key>StandardOutPath</key>
  <string>/tmp/english-vocab-daily-ai-words.log</string>
  <key>StandardErrorPath</key>
  <string>/tmp/english-vocab-daily-ai-words.err</string>
</dict>
</plist>
EOF
```

登録します。

```bash
launchctl load ~/Library/LaunchAgents/com.english-vocab.daily-ai-words.plist
```

すぐに一度試す場合は次を実行します。

```bash
launchctl start com.english-vocab.daily-ai-words
```

停止する場合は次を実行します。

```bash
launchctl unload ~/Library/LaunchAgents/com.english-vocab.daily-ai-words.plist
```

## アプリ起動時に今日分を自動追加する方法

アプリを開いたタイミングで、今日まだAI追加していなければ5語追加することもできます。

```bash
export OPENAI_API_KEY="sk-..."
export AUTO_ADD_AI_WORDS=1
streamlit run app.py
```

## Supabaseで外出先スマホ対応にする方法

外出先でスマホだけで使うには、アプリをStreamlit Community Cloudなどに公開し、単語データをSupabaseに保存します。

### 1. Supabaseプロジェクトを作成

[Supabase](https://supabase.com/) で新しいプロジェクトを作成します。

作成後、Supabaseのダッシュボードで `SQL Editor` を開き、このリポジトリの `supabase_schema.sql` の内容を実行します。

これで次のテーブルが作られます。

- `words`: 単語と学習履歴
- `app_settings`: AI自動追加の最終実行日など

### 2. Supabaseの接続情報を確認

Supabaseの `Project Settings` → `Data API` または `API` で次を確認します。

- Project URL
- `service_role` key

`service_role` key は強い権限を持つ秘密情報です。GitHubや公開ファイルには絶対に書かないでください。

### 3. ローカルでSupabase接続を試す

ローカルでは環境変数で設定できます。

```bash
export SUPABASE_URL="https://your-project.supabase.co"
export SUPABASE_SERVICE_ROLE_KEY="your-service-role-key"
export APP_PASSWORD="好きなパスワード"
```

既存の `words.csv` をSupabaseへ移行します。

```bash
python migrate_csv_to_supabase.py
```

アプリを起動します。

```bash
streamlit run app.py
```

画面上部に `保存先: Supabase` と表示されれば成功です。

### 4. Streamlit Community Cloudに公開

GitHubにこのフォルダをリポジトリとしてアップロードし、[Streamlit Community Cloud](https://streamlit.io/cloud) で `app.py` を指定してデプロイします。

Streamlit Cloudの `Secrets` に次のように設定します。

```toml
SUPABASE_URL = "https://your-project.supabase.co"
SUPABASE_SERVICE_ROLE_KEY = "your-service-role-key"
OPENAI_API_KEY = "sk-..."
OPENAI_MODEL = "gpt-5.4-mini"
APP_PASSWORD = "好きなパスワード"
```

`OPENAI_API_KEY` はAI単語追加を使う場合だけ必要です。`APP_PASSWORD` を設定すると、外から開いた時にパスワード画面が出ます。

### 5. スマホで開く

デプロイ後に発行されるURLをスマホで開きます。

```text
https://your-app-name.streamlit.app
```

ホーム画面に追加しておくと、スマホアプリのように使えます。

## データ保存

ローカルでは初期状態で `words.csv` に保存されます。`SUPABASE_URL` と `SUPABASE_SERVICE_ROLE_KEY` が設定されている場合は、Supabaseの `words` テーブルに保存されます。

主な列は次の通りです。

- `word`: 英単語
- `pronunciation`: 発音メモ
- `part_of_speech`: 品詞
- `meaning_ja`: 日本語の意味
- `example_en`: 英語の例文
- `example_ja`: 例文の日本語訳
- `cloze_examples`: 穴埋め問題用の例文セット（JSON文字列、最大5件）
- `category`: カテゴリ
- `difficulty`: 難易度
- `low_frequency`: 出題頻度を下げるフラグ
- `correct_count`: 正解数
- `wrong_count`: 不正解数
- `last_studied`: 最終学習日
