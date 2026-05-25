# 毎日自動改善PRの運用

このリポジトリには、毎日1回だけ小さな改善PRを作るGitHub Actions workflowがあります。本番アプリをAIが直接書き換えるのではなく、専用ブランチに変更を作り、テストが通った場合だけPRを開きます。

## 仕組み

1. GitHub Actionsが毎日1回起動します。
2. `IMPROVEMENT_BACKLOG.md` から安全な未完了タスクを1つ選びます。
3. `scripts/auto_improve.py` がOpenAI APIで小さな変更案を作ります。
4. smoke testを実行します。
5. テスト成功時だけ `auto-improve/<run_id>` ブランチをpushし、PRを作成します。
6. テスト失敗時、または変更がない時はPRを作りません。

## 必要なGitHub Secrets

GitHubリポジトリの `Settings` → `Secrets and variables` → `Actions` に設定します。

- `OPENAI_API_KEY`: 自動改善を実行するためのOpenAI APIキーです。これが未設定の場合、workflowは動きますが変更を作りません。
- `AUTO_IMPROVE_MODEL`: 任意です。未設定の場合は `gpt-5-mini` を使います。指定したモデルで失敗した場合は、スクリプトが安全な候補モデルに切り替えます。

通常のアプリ用secretsである `SUPABASE_URL`、`SUPABASE_SERVICE_ROLE_KEY`、`APP_PASSWORD` はStreamlit Cloud側に設定します。GitHub Actionsの自動改善には不要です。

## 必要なGitHub Actions権限

PRを自動作成するには、GitHubリポジトリの `Settings` → `Actions` → `General` で次を有効にします。

- `Workflow permissions`: `Read and write permissions`
- `Allow GitHub Actions to create and approve pull requests`: 有効

この設定がOFFの場合、AI変更の生成やテストが成功しても、最後のPR作成だけ失敗します。

## 毎日の確認フロー

1. iPhoneでGitHubアプリ、またはブラウザからリポジトリを開きます。
2. `Pull requests` を開き、`Auto improvement:` で始まるPRを確認します。
3. PR本文の「改善内容」「なぜ安全か」「実行したテスト結果」「人間が確認すべき点」を読みます。
4. 変更ファイルを確認します。
5. 問題なければ `Merge pull request` を押します。
6. 迷う場合はマージせず、PRにコメントするか閉じます。

## 自動改善の対象

- スマホUIの微調整
- 学習体験の小改善
- smoke testや単体テストの追加
- 読みやすさ改善の小さなリファクタ
- 軽微な表示バグ修正

## 自動改善の対象外

- Supabaseテーブル削除、列削除、データ削除
- `supabase_schema.sql` の破壊的変更
- RLS、認証、権限設計の大きな変更
- service role key、API key、passwordなどsecretsの変更
- 課金、外部サービス契約、料金に関わる変更
- 本番データを直接書き換える処理

## テスト

ローカルでは次を実行します。

```bash
python -B -m unittest discover -s tests
```

アプリ起動の確認は次です。

```bash
streamlit run app.py
```

## 止め方

一時停止したい場合は、GitHubの `Actions` → `Daily Auto Improvement` → `Disable workflow` を押します。

完全に止めたい場合は `.github/workflows/daily-auto-improvement.yml` を削除するPRを作ってマージします。

## トラブル時

- PRが作られない場合: `Actions` の実行ログを確認します。`OPENAI_API_KEY` が未設定だと変更は作られません。
- テストで失敗する場合: 失敗ログを確認し、手動で修正します。失敗時はPRは作られません。
- 変更が大きすぎる場合: PRをマージせず閉じて、`IMPROVEMENT_BACKLOG.md` のタスクをより小さく分けます。
- 不安な変更が出た場合: PRを閉じます。本番ブランチには反映されません。
