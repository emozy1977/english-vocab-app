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

    def test_cloze_examples_for_values_uses_natural_primary_example_only(self) -> None:
        examples = app.cloze_examples_for_values(
            {
                "word": "manage",
                "part_of_speech": "verb",
                "meaning_ja": "管理する",
                "example_en": "We manage the project carefully.",
                "example_ja": "私たちはそのプロジェクトを慎重に管理します。",
            }
        )

        self.assertEqual(
            examples,
            [
                {
                    "en": "We manage the project carefully.",
                    "ja": "私たちはそのプロジェクトを慎重に管理します。",
                }
            ],
        )

    def test_cloze_examples_filter_old_pedagogical_templates(self) -> None:
        examples = app.cloze_examples_for_values(
            {
                "word": "mitigate",
                "part_of_speech": "verb",
                "meaning_ja": "軽減する、和らげる",
                "example_en": "",
                "example_ja": "",
                "cloze_examples": app.encode_cloze_examples(
                    [
                        {"en": "The form mitigates is used with he, she, or it.", "ja": "三単現の形を入れます。"},
                    ]
                ),
            }
        )
        text = " ".join(example["en"] for example in examples)

        self.assertNotIn("The form", text)
        self.assertEqual(examples, [])

    def test_cloze_examples_keep_valid_stored_business_examples(self) -> None:
        examples = app.cloze_examples_for_values(
            {
                "word": "mitigate",
                "part_of_speech": "verb",
                "meaning_ja": "軽減する、和らげる",
                "example_en": "",
                "example_ja": "",
                "cloze_examples": app.encode_cloze_examples(
                    [
                        {"en": "The release checklist mitigates deployment risk.", "ja": "リリースチェックリストはデプロイ時のリスクを軽減します。"},
                        {"en": "The form mitigates is used with he, she, or it.", "ja": "三単現の形を入れます。"},
                    ]
                ),
            }
        )

        self.assertEqual(
            examples,
            [
                {
                    "en": "The release checklist mitigates deployment risk.",
                    "ja": "リリースチェックリストはデプロイ時のリスクを軽減します。",
                }
            ],
        )

    def test_encode_cloze_examples_deduplicates_and_limits_to_five(self) -> None:
        encoded = app.encode_cloze_examples(
            [
                {"en": "She manages the project.", "ja": "彼女はプロジェクトを管理します。"},
                {"en": "She manages the project.", "ja": "重複"},
                {"en": "We manage the project.", "ja": ""},
                {"en": "They managed the project.", "ja": ""},
                {"en": "The project was managed well.", "ja": ""},
                {"en": "Managing the project takes time.", "ja": ""},
                {"en": "Extra example.", "ja": ""},
            ]
        )

        parsed = app.parse_cloze_examples(encoded)

        self.assertEqual(len(parsed), 5)
        self.assertEqual(parsed[0]["ja"], "彼女はプロジェクトを管理します。")

    def test_answer_diff_html_highlights_spelling_mistakes(self) -> None:
        html = app.answer_diff_html("implement", "implment")

        self.assertIn("diff-missing", html)
        self.assertIn("[e]", html)

    def test_answer_diff_html_escapes_user_input(self) -> None:
        html = app.answer_diff_html("test", "<test>")

        self.assertIn("&lt;", html)
        self.assertNotIn("<test>", html)

    def test_normalize_sentence_answer_ignores_case_punctuation_and_spacing(self) -> None:
        self.assertEqual(
            app.normalize_sentence_answer("  The team, will   implement it. "),
            app.normalize_sentence_answer("the team will implement it"),
        )

    def test_today_uses_japan_timezone_by_default(self) -> None:
        self.assertEqual(app.DEFAULT_TIMEZONE, "Asia/Tokyo")

    def test_tts_cache_path_is_stable_and_safe(self) -> None:
        path = app.tts_cache_path("The team will implement it.", "gpt-4o-mini-tts", "nova")

        self.assertTrue(path.startswith("gpt-4o-mini-tts/nova/"))
        self.assertTrue(path.endswith(".mp3"))
        self.assertNotIn("The team", path)

    def test_expected_tts_cache_paths_lists_unique_cloze_sentences(self) -> None:
        df = pd.DataFrame(
            [
                [
                    1,
                    "manage",
                    "",
                    "verb",
                    "管理する",
                    "",
                    "",
                    app.encode_cloze_examples(
                        [
                            {"en": "She manages the project.", "ja": "彼女はプロジェクトを管理します。"},
                            {"en": "She manages the project.", "ja": "重複"},
                            {"en": "We manage the release plan.", "ja": "私たちはリリース計画を管理します。"},
                        ]
                    ),
                    "Business",
                    "3",
                    False,
                    0,
                    0,
                    "",
                ],
            ],
            columns=app.COLUMNS,
        )

        paths = app.expected_tts_cache_paths(app.normalize_df(df), "gpt-4o-mini-tts", "nova")

        self.assertEqual(paths["text"].tolist(), ["She manages the project.", "We manage the release plan."])
        self.assertTrue(paths["path"].str.endswith(".mp3").all())

    def test_priority_uses_wrong_minus_correct_as_weakness_score(self) -> None:
        df = pd.DataFrame(
            [
                [1, "known", "", "other", "既知", "", "", "[]", "Test", "3", False, 5, 1, "2026-05-20"],
                [2, "weak", "", "other", "苦手", "", "", "[]", "Test", "3", False, 1, 5, "2026-05-20"],
            ],
            columns=app.COLUMNS,
        )

        ordered = app.priority(app.normalize_df(df))

        self.assertEqual(ordered.iloc[0]["word"], "weak")
        self.assertEqual(ordered.iloc[0]["weakness_score"], 4)

    def test_dashboard_stats_summarizes_learning_progress(self) -> None:
        df = pd.DataFrame(
            [
                [1, "mastered", "", "other", "定着", "", "", "[]", "Test", "3", False, 3, 0, "2026-06-01"],
                [2, "weak", "", "other", "苦手", "", "", "[]", "Test", "3", False, 1, 3, "2026-05-31"],
                [3, "known", "", "other", "既知", "", "", "[]", "Test", "3", False, 1, 0, "2026-05-30"],
                [4, "new", "", "other", "新規", "", "", "[]", "Test", "3", False, 0, 0, ""],
            ],
            columns=app.COLUMNS,
        )

        stats = app.dashboard_stats(df, today_value="2026-06-01", daily_goal=5)

        self.assertEqual(stats["total_words"], 4)
        self.assertEqual(stats["today_count"], 1)
        self.assertEqual(stats["weak_count"], 1)
        self.assertEqual(stats["new_count"], 1)
        self.assertEqual(stats["mastered_count"], 1)
        self.assertEqual(stats["streak"], 3)
        self.assertAlmostEqual(stats["accuracy"], 5 / 8)

    def test_dashboard_default_daily_goal_is_fifty_answers(self) -> None:
        stats = app.dashboard_stats(pd.DataFrame(app.SAMPLE_WORDS, columns=app.COLUMNS), today_value="2026-06-01")

        self.assertEqual(stats["daily_goal"], 50)

    def test_dashboard_stats_uses_study_events_for_today_count(self) -> None:
        df = pd.DataFrame(
            [
                [1, "alpha", "", "other", "A", "", "", "[]", "Test", "3", False, 1, 0, "2026-06-01"],
                [2, "beta", "", "other", "B", "", "", "[]", "Test", "3", False, 0, 1, "2026-06-01"],
            ],
            columns=app.COLUMNS,
        )
        events = pd.DataFrame(
            [
                {"word_id": 1, "word": "alpha", "mode": "written", "correct": True, "studied_on": "2026-06-01", "studied_at": "2026-06-01T08:00:00+09:00"},
                {"word_id": 2, "word": "beta", "mode": "fill", "correct": False, "studied_on": "2026-05-31", "studied_at": "2026-05-31T23:00:00+09:00"},
            ]
        )

        stats = app.dashboard_stats(df, events=events, today_value="2026-06-01", daily_goal=5)

        self.assertEqual(stats["today_count"], 1)
        self.assertEqual(stats["today_correct"], 1)
        self.assertEqual(stats["today_wrong"], 0)
        self.assertTrue(stats["event_log_available"])

    def test_daily_event_counts_fills_missing_days(self) -> None:
        events = pd.DataFrame(
            [
                {"word_id": 1, "word": "alpha", "mode": "written", "correct": True, "studied_on": "2026-05-30", "studied_at": "2026-05-30T08:00:00+09:00"},
                {"word_id": 2, "word": "beta", "mode": "fill", "correct": False, "studied_on": "2026-06-01", "studied_at": "2026-06-01T08:00:00+09:00"},
                {"word_id": 3, "word": "gamma", "mode": "study", "correct": True, "studied_on": "2026-06-01", "studied_at": "2026-06-01T09:00:00+09:00"},
            ]
        )

        counts = app.daily_event_counts(events, today_value="2026-06-01", days=4)

        self.assertEqual(counts["日付"].tolist(), ["2026-05-29", "2026-05-30", "2026-05-31", "2026-06-01"])
        self.assertEqual(counts["学習回数"].tolist(), [0, 1, 0, 2])

    def test_consecutive_learning_days_keeps_yesterday_streak_visible(self) -> None:
        streak = app.consecutive_learning_days({"2026-05-30", "2026-05-31"}, today_value="2026-06-01")

        self.assertEqual(streak, 2)

    def test_weak_words_filters_only_positive_weakness_scores(self) -> None:
        df = pd.DataFrame(
            [
                [1, "strong", "", "other", "得意", "", "", "[]", "Test", "3", False, 5, 1, "2026-05-20"],
                [2, "equal", "", "other", "同じ", "", "", "[]", "Test", "3", False, 2, 2, "2026-05-20"],
                [3, "weak", "", "other", "苦手", "", "", "[]", "Test", "3", False, 1, 4, "2026-05-20"],
            ],
            columns=app.COLUMNS,
        )

        weak = app.weak_words(app.normalize_df(df))

        self.assertEqual(weak["word"].tolist(), ["weak"])
        self.assertEqual(int(weak.iloc[0]["weakness_score"]), 3)

    def test_mixed_ids_includes_difficult_new_and_regular_words(self) -> None:
        df = pd.DataFrame(
            [
                [1, "regular", "", "other", "普通", "", "", "[]", "Test", "3", False, 2, 1, "2026-05-20"],
                [2, "difficult", "", "other", "苦手", "", "", "[]", "Test", "3", False, 1, 4, "2026-05-20"],
                [3, "new", "", "other", "新規", "", "", "[]", "Test", "3", False, 0, 0, ""],
            ],
            columns=app.COLUMNS,
        )

        ids = app.mixed_ids(app.normalize_df(df))

        self.assertEqual(ids[:2], [3, 2])
        self.assertCountEqual(ids, [1, 2, 3])

    def test_mixed_ids_places_low_frequency_words_later(self) -> None:
        df = pd.DataFrame(
            [
                [1, "new_low", "", "other", "新規低頻度", "", "", "[]", "Test", "3", True, 0, 0, ""],
                [2, "new_normal", "", "other", "新規", "", "", "[]", "Test", "3", False, 0, 0, ""],
                [3, "weak", "", "other", "苦手", "", "", "[]", "Test", "3", False, 1, 4, "2026-05-20"],
            ],
            columns=app.COLUMNS,
        )

        ids = app.mixed_ids(app.normalize_df(df))

        self.assertEqual(ids, [2, 3, 1])

    def test_next_id_skips_recent_low_frequency_word_when_possible(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "new_low", "", "other", "新規低頻度", "", "", "[]", "Test", "3", True, 0, 0, ""],
                    [2, "new_normal", "", "other", "新規", "", "", "[]", "Test", "3", False, 0, 0, ""],
                    [3, "weak", "", "other", "苦手", "", "", "[]", "Test", "3", False, 1, 4, "2026-05-20"],
                ],
                columns=app.COLUMNS,
            )
        )

        self.assertEqual(app.next_id(df, current=3, recent_ids=[1, 2, 3]), 2)

    def test_next_id_keeps_all_low_frequency_words_on_cooldown(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "low_recent", "", "other", "低頻度1", "", "", "[]", "Test", "3", True, 0, 0, ""],
                    [2, "low_other", "", "other", "低頻度2", "", "", "[]", "Test", "3", True, 0, 0, ""],
                    [3, "normal_a", "", "other", "通常1", "", "", "[]", "Test", "3", False, 0, 0, ""],
                    [4, "normal_b", "", "other", "通常2", "", "", "[]", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )

        self.assertEqual(app.next_id(df, current=4, recent_ids=[1, 3, 4]), 3)

    def test_next_id_allows_low_frequency_after_cooldown(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "low", "", "other", "低頻度", "", "", "[]", "Test", "3", True, 0, 0, ""],
                    [2, "normal_a", "", "other", "通常1", "", "", "[]", "Test", "3", False, 0, 0, ""],
                    [3, "normal_b", "", "other", "通常2", "", "", "[]", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )

        self.assertEqual(app.next_id(df, current=2, recent_ids=[2, 3]), 1)

    def test_next_id_allows_low_frequency_when_no_other_choice(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "only_low", "", "other", "低頻度だけ", "", "", "[]", "Test", "3", True, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )

        self.assertEqual(app.next_id(df, current=1, recent_ids=[1]), 1)

    def test_mixed_ids_places_many_first_try_correct_words_later(self) -> None:
        df = pd.DataFrame(
            [
                [1, "easy", "", "other", "簡単", "", "", "[]", "Test", "3", False, 8, 0, "2026-05-20"],
                [2, "less_known", "", "other", "まだ浅い", "", "", "[]", "Test", "3", False, 1, 0, "2026-05-20"],
                [3, "new", "", "other", "新規", "", "", "[]", "Test", "3", False, 0, 0, ""],
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
                    [1, "one", "", "other", "1", "", "", "[]", "Test", "3", False, 0, 0, ""],
                    [3, "three", "", "other", "3", "", "", "[]", "Test", "3", False, 0, 0, ""],
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
                    [1, "target", "", "other", "対象", "", "", "[]", "Test", "3", False, 0, 0, ""],
                ],
                columns=app.COLUMNS,
            )
        )
        saved_frames: list[pd.DataFrame] = []

        with (
            patch.object(app, "supabase_enabled", return_value=False),
            patch.object(app, "save_words", side_effect=lambda frame: saved_frames.append(frame.copy())),
            patch.object(app, "set_words", side_effect=lambda frame: frame),
            patch.object(app, "record_study_event", return_value=True),
        ):
            updated = app.update_stats(df, 1, True, "written")

        self.assertEqual(int(updated.loc[0, "correct_count"]), 1)
        self.assertEqual(int(saved_frames[0].loc[0, "correct_count"]), 1)

    def test_study_event_from_row_records_japan_date_and_mode(self) -> None:
        row = pd.Series({"id": 7, "word": "implement"})

        with patch.object(app, "today", return_value="2026-06-01"):
            event = app.study_event_from_row(row, "listening", False)

        self.assertEqual(event["word_id"], 7)
        self.assertEqual(event["word"], "implement")
        self.assertEqual(event["mode"], "listening")
        self.assertFalse(event["correct"])
        self.assertEqual(event["studied_on"], "2026-06-01")

    def test_update_low_frequency_saves_local_dataframe(self) -> None:
        df = app.normalize_df(
            pd.DataFrame(
                [
                    [1, "target", "", "other", "対象", "", "", "[]", "Test", "3", False, 0, 0, ""],
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
                    [1, "target", "", "other", "対象", "", "", "[]", "Test", "3", False, 0, 0, ""],
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
