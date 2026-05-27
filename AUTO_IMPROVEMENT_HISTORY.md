# 自動改善履歴

マージ済みの自動改善PRで扱ったタスクを記録します。自動改善スクリプトはこの履歴と `IMPROVEMENT_BACKLOG.md` を見て、最近扱ったタスクを避けます。
- 2026-05-26: 学習カードで「次のカード」後に状態が分かりやすい表示を追加する。 — 重複しやすかったため、今後の自動選択から外しました。
- 2026-05-26: 学習統計更新処理の重複を減らす。 — Reduce duplication in study-stat update by centralizing persistence into save_stats and always using it from update_stats. When not using Supabase, save_stats now updates only the relevant row in the CSV (or appends if missing) instead of saving session_state; Supabase behavior is unchanged. update_stats now unconditionally calls save_stats, reducing duplicated save logic.
- 2026-05-27: ボタン文言を短くし、スマホで折り返しにくくする。 — Shorten several button labels to avoid wrapping on small screens (スマホ). Updated labels: '保存する'->'保存', '意味を表示'->'表示', '次のカード'/'次の問題'->'次へ', '判定する'->'判定', 'AIで今日分を追加'->'AI追加', 'AIで品詞を補完'->'品詞補完', pronunciation button '発音を聞く'->'発音'. Kept changes minimal and UI-only.
