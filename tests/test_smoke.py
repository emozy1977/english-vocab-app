from __future__ import annotations

import unittest

import pandas as pd

import app


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


if __name__ == "__main__":
    unittest.main()
