# 自動改善履歴

マージ済みの自動改善PRで扱ったタスクを記録します。自動改善スクリプトはこの履歴と `IMPROVEMENT_BACKLOG.md` を見て、最近扱ったタスクを避けます。
- 2026-05-26: 学習カードで「次のカード」後に状態が分かりやすい表示を追加する。 — 重複しやすかったため、今後の自動選択から外しました。
- 2026-05-26: 新規追加単語が学習画面に出る理由をUI上で短く示す。 — Show a short UI hint on study cards for newly added words. A small "新規" pill is added to the card when a word has never been studied (correct_count + wrong_count == 0), with a title explaining briefly why it appears in study. Added CSS for the new pill style.
