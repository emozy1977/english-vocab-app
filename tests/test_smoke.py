from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import app
import scripts.auto_improve as auto_improve


class AppSmokeTests(unittest.TestCase):
    def test_normalize_df_adds_missing_columns_and_defaults(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "word": "sample",
                    "meaning_ja": "例",
                    "example_en": "This is a sample.",
                }
            ]
        )

        normalized = app.normalize_df(df)

        self.assertEqual(list(normalized.columns), app.COLUMNS)
        self.assertEqual(normalized.loc[0, "id"], 1)
        self.assertEqual(normalized.loc[0, "difficulty"], "3")
        self.assertEqual(normalized.loc[0, "part_of_speech"], "other")
        self.assertEqual(normalized.loc[0, "correct_count"], 0)
        self.assertEqual(normalized.loc[0, "wrong_count"], 0)

    def test_blank_sentence_replaces_word_case_insensitively(self) -> None:
        sentence = "We need to Implement the plan quickly."

        self.assertEqual(
            app.blank_sentence(sentence, "implement"),
            "We need to _____ the plan quickly.",
        )

    def test_priority_uses_wrong_minus_correct_as_weakness_score(self) -> None:
        df = pd.DataFrame(
            [
                [1, "known", "", "other", "既知", "", "", "Test", "3", 5, 1, "2026-05-20"],
                [2, "weak", "", "other", "苦手", "", "", "Test", "3", 1, 5, "2026-05-20"],
            ],
            columns=app.COLUMNS,
        )

        ordered = app.priority(app.normalize_df(df))

        self.assertEqual(ordered.iloc[0]["word"], "weak")
        self.assertEqual(ordered.iloc[0]["weakness_score"], 4)

    def test_mixed_ids_includes_difficult_new_and_regular_words(self) -> None:
        df = pd.DataFrame(
            [
                [1, "regular", "", "other", "普通", "", "", "Test", "3", 2, 1, "2026-05-20"],
                [2, "difficult", "", "other", "苦手", "", "", "Test", "3", 1, 4, "2026-05-20"],
                [3, "new", "", "other", "新規", "", "", "Test", "3", 0, 0, ""],
            ],
            columns=app.COLUMNS,
        )

        ids = app.mixed_ids(app.normalize_df(df))

        self.assertEqual(ids[0], 2)
        self.assertIn(3, ids[:2])
        self.assertCountEqual(ids, [1, 2, 3])

    def test_auto_improve_skips_recent_history_tasks(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            backlog = root / "IMPROVEMENT_BACKLOG.md"
            history = root / "AUTO_IMPROVEMENT_HISTORY.md"
            backlog.write_text(
                "- [ ] First safe task\n"
                "- [ ] Second safe task\n",
                encoding="utf-8",
            )
            history.write_text("- 2026-05-25: First safe task\n", encoding="utf-8")

            self.assertEqual(auto_improve.pick_task(backlog, history), "Second safe task")

    def test_auto_improve_marks_task_done_and_records_history(self) -> None:
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            backlog = root / "IMPROVEMENT_BACKLOG.md"
            history = root / "AUTO_IMPROVEMENT_HISTORY.md"
            task = "Improve a small label"
            backlog.write_text(f"- [ ] {task}\n", encoding="utf-8")
            result = auto_improve.ImprovementResult(
                summary="Changed one label.",
                safety="UI text only.",
                tests=["not run"],
                human_check=["read the label"],
                files=[],
            )

            auto_improve.mark_task_done(backlog, task)
            auto_improve.append_history(history, task, result)

            self.assertIn(f"- [x] {task}", backlog.read_text(encoding="utf-8"))
            self.assertIn(task, history.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
