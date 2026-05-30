from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

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
        self.assertFalse(bool(normalized.loc[0, "low_frequency"]))
        self.assertEqual(normalized.loc[0, "correct_count"], 0)
        self.assertEqual(normalized.loc[0, "wrong_count"], 0)

    def test_blank_sentence_replaces_word_case_insensitively(self) -> None:
        sentence = "We need to Implement the plan quickly."

        self.assertEqual(
            app.blank_sentence(sentence, "implement"),
            "We need to _____ the plan quickly.",
        )

    def test_blank_sentence_replaces_basic_inflected_forms(self) -> None:
        self.assertEqual(
            app.blank_sentence("We refined the proposal after several meetings.", "refine"),
            "We _____ the proposal after several meetings.",
        )
        self.assertEqual(
            app.blank_sentence("The team is consolidating reports.", "consolidate"),
            "The team is _____ reports.",
        )

    def test_blank_sentence_and_answer_returns_inflected_answer(self) -> None:
        prompt, answer = app.blank_sentence_and_answer("She manages the project well.", "manage")

        self.assertEqual(prompt, "She _____ the project well.")
        self.assertEqual(answer, "manages")

    def test_blank_sentence_and_answer_falls_back_to_base_word(self) -> None:
        prompt, answer = app.blank_sentence_and_answer("She leads the project well.", "manage")

        self.assertEqual(prompt, "_____ She leads the project well.")
        self.assertEqual(answer, "manage")

    def test_answer_diff_html_highlights_spelling_mistakes(self) -> None:
        html = app.answer_diff_html("implement", "implment")

        self.assertIn("diff-missing", html)
        self.assertIn("[e]", html)

    def test_answer_diff_html_escapes_user_input(self) -> None:
        html = app.answer_diff_html("test", "<test>")

        self.assertIn("&lt;", html)
        self.assertNotIn("<test>", html)

    def test_priority_uses_wrong_minus_correct_as_weakness_score(self) -> None:
        df = pd.DataFrame(
            [
                [1, "known", "", "other", "既知", "", "", "Test", "3", False, 5, 1, "2026-05-20"],
                [2, "weak", "", "other", "苦手", "", "", "Test", "3", False, 1, 5, "2026-05-20"],
            ],
            columns=app.COLUMNS,
        )

        ordered = app.priority(app.normalize_df(df))

        self.assertEqual(ordered.iloc[0]["word"], "weak")
        self.assertEqual(ordered.iloc[0]["weakness_score"], 4)

    def test_mixed_ids_includes_difficult_new_and_regular_words(self) -> None:
        df = pd.DataFrame(
            [
                [1, "regular", "", "other", "普通", "", "", "Test", "3", False, 2, 1, "2026-05-20"],
                [2, "difficult", "", "other", "苦手", "", "", "Test", "3", False, 1, 4, "2026-05-20"],
                [3, "new", "", "other", "新規", "", "", "Test", "3", False, 0, 0, ""],
            ],
            columns=app.COLUMNS,
        )

        ids = app.mixed_ids(app.normalize_df(df))

        self.assertEqual(ids[:2], [3, 2])
        self.assertCountEqual(ids, [1, 2, 3])

    def test_mixed_ids_places_low_frequency_words_later(self) -> None:
        df = pd.DataFrame(
            [
                [1, "new_low", "", "other", "新規低頻度", "", "", "Test", "3", True, 0, 0, ""],
                [2, "new_normal", "", "other", "新規", "", "", "Test", "3", False, 0, 0, ""],
                [3, "weak", "", "other", "苦手", "", "", "Test", "3", False, 1, 4, "2026-05-20"],
            ],
            columns=app.COLUMNS,
        )

        ids = app.mixed_ids(app.normalize_df(df))

        self.assertEqual(ids, [2, 3, 1])

    def test_next_id_skips_recent_low_frequency_word_when_possible(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "new_low", "", "other", "新規低頻度", "", "", "Test", "3", True, 0, 0, ""],
                    [2, "new_normal", "", "other", "新規", "", "", "Test", "3", False, 0, 0, ""],
                    [3, "weak", "", "other", "苦手", "", "", "Test", "3", False, 1, 4, "2026-05-20"],
                ],
                columns=app.COLUMNS,
            )
        )

        self.assertEqual(app.next_id(df, current=3, recent_ids=[1, 2, 3]), 2)

    def test_next_id_keeps_all_low_frequency_words_on_cooldown(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "low_recent", "", "other", "低頻度1", "", "", "Test", "3", True, 0, 0, ""],
                    [2, "low_other", "", "other", "低頻度2", "", "", "Test", "3", True, 0, 0, ""],
                    [3, "normal_a", "", "other", "通常1", "", "", "Test", "3", False, 0, 0, ""],
                    [4, "normal_b", "", "other", "通常2", "", "", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )

        self.assertEqual(app.next_id(df, current=4, recent_ids=[1, 3, 4]), 3)

    def test_next_id_allows_low_frequency_after_cooldown(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "low", "", "other", "低頻度", "", "", "Test", "3", True, 0, 0, ""],
                    [2, "normal_a", "", "other", "通常1", "", "", "Test", "3", False, 0, 0, ""],
                    [3, "normal_b", "", "other", "通常2", "", "", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )

        self.assertEqual(app.next_id(df, current=2, recent_ids=[2, 3]), 1)

    def test_next_id_allows_low_frequency_when_no_other_choice(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "only_low", "", "other", "低頻度だけ", "", "", "Test", "3", True, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )

        self.assertEqual(app.next_id(df, current=1, recent_ids=[1]), 1)

    def test_mixed_ids_places_many_first_try_correct_words_later(self) -> None:
        df = pd.DataFrame(
            [
                [1, "easy", "", "other", "簡単", "", "", "Test", "3", False, 8, 0, "2026-05-20"],
                [2, "less_known", "", "other", "まだ浅い", "", "", "Test", "3", False, 1, 0, "2026-05-20"],
                [3, "new", "", "other", "新規", "", "", "Test", "3", False, 0, 0, ""],
            ],
            columns=app.COLUMNS,
        )

        ids = app.mixed_ids(app.normalize_df(df))

        self.assertEqual(ids, [3, 2, 1])

    def test_is_first_quiz_attempt_only_counts_first_answer(self) -> None:
        self.assertTrue(app.is_first_quiz_attempt(None, 1))
        self.assertTrue(app.is_first_quiz_attempt({"id": 2, "correct": False}, 1))
        self.assertFalse(app.is_first_quiz_attempt({"id": 1, "correct": False}, 1))

    def test_pushed_history_moves_word_to_end(self) -> None:
        self.assertEqual(app.pushed_history([1, 2, 1], 2), [1, 2])

    def test_pop_previous_id_skips_missing_words(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "one", "", "other", "1", "", "", "Test", "3", False, 0, 0, ""],
                    [3, "three", "", "other", "3", "", "", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )

        previous_id, remaining = app.pop_previous_id([1, 2, 3], df)

        self.assertEqual(previous_id, 3)
        self.assertEqual(remaining, [1, 2])

    def test_update_stats_saves_updated_local_dataframe(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "target", "", "other", "対象", "", "", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )
        saved_frames: list[pd.DataFrame] = []

        with (
            patch.object(app, "supabase_enabled", return_value=False),
            patch.object(app, "save_words", side_effect=lambda frame: saved_frames.append(frame.copy())),
            patch.object(app, "set_words", side_effect=lambda frame: frame),
        ):
            updated = app.update_stats(df, 1, True)

        self.assertEqual(int(updated.loc[0, "correct_count"]), 1)
        self.assertEqual(int(saved_frames[0].loc[0, "correct_count"]), 1)

    def test_update_low_frequency_saves_local_dataframe(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "target", "", "other", "対象", "", "", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )
        saved_frames: list[pd.DataFrame] = []

        with (
            patch.object(app, "supabase_enabled", return_value=False),
            patch.object(app, "save_words", side_effect=lambda frame: saved_frames.append(frame.copy())),
            patch.object(app, "set_words", side_effect=lambda frame: frame),
        ):
            updated = app.update_low_frequency(df, 1, True)

        self.assertTrue(bool(updated.loc[0, "low_frequency"]))
        self.assertTrue(bool(saved_frames[0].loc[0, "low_frequency"]))

    def test_update_low_frequency_reports_missing_supabase_column(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "target", "", "other", "対象", "", "", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )

        class BrokenQuery:
            def update(self, values):
                return self

            def eq(self, column, value):
                return self

            def execute(self):
                raise RuntimeError("Could not find the 'low_frequency' column")

        class BrokenClient:
            def table(self, name):
                return BrokenQuery()

        with (
            patch.object(app, "supabase_enabled", return_value=True),
            patch.object(app, "supabase_client", return_value=BrokenClient()),
        ):
            with self.assertRaises(app.LowFrequencySaveError):
                app.update_low_frequency(df, 1, True)

    def test_saved_ai_category_uses_local_marker(self) -> None:
        with TemporaryDirectory() as tmp:
            with (
                patch.object(app, "supabase_enabled", return_value=False),
                patch("app.Path", side_effect=lambda value: Path(tmp) / value),
            ):
                app.set_saved_ai_category("Business")

                self.assertEqual(app.saved_ai_category(), "Business")

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

            self.assertEqual(auto_improve.pick_task(backlog, history, offset=0), "Second safe task")

    def test_auto_improve_reads_history_task_without_summary_suffix(self) -> None:
        with TemporaryDirectory() as tmp:
            history = Path(tmp) / "AUTO_IMPROVEMENT_HISTORY.md"
            history.write_text(
                "- 2026-05-26: First safe task — Already handled.\n",
                encoding="utf-8",
            )

            self.assertEqual(auto_improve.recent_history_tasks(history), {"First safe task"})

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
