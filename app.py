from __future__ import annotations

import html
import json
import os
import re
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DATA_FILE = Path(__file__).with_name("words.csv")
COLUMNS = ["id", "word", "pronunciation", "part_of_speech", "meaning_ja", "example_en", "example_ja", "category", "difficulty", "low_frequency", "correct_count", "wrong_count", "last_studied"]
COUNT_COLUMNS = ["correct_count", "wrong_count"]
BOOL_COLUMNS = ["low_frequency"]
SUPABASE_TABLE = "words"
SUPABASE_SETTINGS_TABLE = "app_settings"
DEFAULT_AI_MODEL = "gpt-5.4-mini"
LOW_FREQUENCY_GAP = 20
SESSION_WORDS_KEY = "words_df"
PARTS_OF_SPEECH = ["noun", "verb", "adjective", "adverb", "phrase", "other"]
PARTS_OF_SPEECH_SET = set(PARTS_OF_SPEECH)

SAMPLE_WORDS = [
    [1, "incorporate", "in-KOR-puh-rayt", "verb", "取り入れる、組み込む", "We need to incorporate user feedback into the next version.", "次のバージョンにユーザーの意見を取り入れる必要があります。", "Business", "4", False, 0, 0, ""],
    [2, "consolidate", "kun-SOL-ih-dayt", "verb", "統合する、強化する", "The team will consolidate multiple reports into one dashboard.", "チームは複数のレポートを1つのダッシュボードに統合します。", "Business", "4", False, 0, 0, ""],
    [3, "appropriate", "uh-PROH-pree-uht", "adjective", "適切な", "Please choose the most appropriate response for the situation.", "その状況に最も適切な返答を選んでください。", "Academic", "3", False, 0, 0, ""],
    [4, "implement", "IM-pluh-ment", "verb", "実行する、実装する", "The company plans to implement a new training program.", "会社は新しい研修プログラムを実施する予定です。", "Business", "3", False, 0, 0, ""],
    [5, "overlook", "oh-ver-LOOK", "verb", "見落とす、大目に見る", "It is easy to overlook small errors when you are tired.", "疲れていると小さな誤りを見落としやすいです。", "Daily", "3", False, 0, 0, ""],
    [6, "fatigue", "fuh-TEEG", "noun", "疲労", "Long meetings can cause mental fatigue.", "長い会議は精神的な疲労を引き起こすことがあります。", "Health", "2", False, 0, 0, ""],
    [7, "retention", "ree-TEN-shun", "noun", "保持、定着", "Regular review improves vocabulary retention.", "定期的な復習は語彙の定着を高めます。", "Learning", "4", False, 0, 0, ""],
    [8, "elaborate", "ih-LAB-uh-rayt", "verb", "詳しく説明する", "Could you elaborate on your main idea?", "主な考えについて詳しく説明してもらえますか。", "Academic", "3", False, 0, 0, ""],
    [9, "conversely", "KON-ver-slee", "adverb", "反対に、逆に", "Some tasks require speed; conversely, others require careful planning.", "速さが必要な作業もありますが、逆に慎重な計画が必要な作業もあります。", "Academic", "4", False, 0, 0, ""],
    [10, "recurrent", "ree-KUR-unt", "adjective", "繰り返し起こる", "The app helps users review recurrent mistakes.", "そのアプリはユーザーが繰り返し起こる間違いを復習するのに役立ちます。", "Academic", "4", False, 0, 0, ""],
]


class LowFrequencySaveError(RuntimeError):
    pass


def config(key: str, default: str = "") -> str:
    value = os.getenv(key, "").strip()
    if value:
        return value
    local_secret_paths = [
        Path.home() / ".streamlit" / "secrets.toml",
        Path.cwd() / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
    ]
    is_streamlit_cloud = bool(os.getenv("STREAMLIT_CLOUD")) or Path.home().name == "adminuser" or Path("/mount/src").exists()
    if not any(path.exists() for path in local_secret_paths) and not is_streamlit_cloud:
        return default
    try:
        return str(st.secrets.get(key, default)).strip()
    except Exception:
        return default


def supabase_enabled() -> bool:
    return bool(config("SUPABASE_URL") and (config("SUPABASE_SERVICE_ROLE_KEY") or config("SUPABASE_KEY")))


def supabase_client():
    from supabase import create_client

    key = config("SUPABASE_SERVICE_ROLE_KEY") or config("SUPABASE_KEY")
    return create_client(config("SUPABASE_URL"), key)


def normalize_pos(value: object) -> str:
    value = str(value).strip().lower()
    return value if value in PARTS_OF_SPEECH_SET else "other"


def normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on", "checked"}


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = False if col in BOOL_COLUMNS else 0 if col in COUNT_COLUMNS else ""
    df = df[COLUMNS].fillna("")
    if df.empty:
        df = pd.DataFrame(SAMPLE_WORDS, columns=COLUMNS)
    df["id"] = pd.to_numeric(df["id"], errors="coerce")
    next_word_id = int(df["id"].max()) + 1 if df["id"].notna().any() else 1
    for idx in df[df["id"].isna()].index:
        df.at[idx, "id"] = next_word_id
        next_word_id += 1
    df["id"] = df["id"].astype(int)
    for col in COUNT_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    df["difficulty"] = df["difficulty"].astype(str).replace({"": "3"})
    df["part_of_speech"] = df["part_of_speech"].apply(normalize_pos)
    for col in BOOL_COLUMNS:
        df[col] = df[col].apply(normalize_bool)
    return df


def save_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    normalized = normalize_df(pd.DataFrame(rows))
    if supabase_enabled():
        supabase_client().table(SUPABASE_TABLE).upsert(normalized[COLUMNS].to_dict("records"), on_conflict="word").execute()
        return
    current = load_words()
    merged = pd.concat([current, normalized], ignore_index=True).drop_duplicates(subset=["word"], keep="last")
    merged.to_csv(DATA_FILE, index=False)


def save_words(df: pd.DataFrame) -> None:
    df = normalize_df(df)
    if supabase_enabled():
        supabase_client().table(SUPABASE_TABLE).upsert(df[COLUMNS].to_dict("records"), on_conflict="word").execute()
        return
    df.to_csv(DATA_FILE, index=False)


def save_stats(row: pd.Series) -> None:
    # Centralized persistence for a single-row stats update.
    # For Supabase: update only the necessary fields.
    # For CSV: load current file, update the matching row (by id) and save the CSV.
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    if supabase_enabled():
        supabase_client().table(SUPABASE_TABLE).update({
            "correct_count": int(row_dict.get("correct_count", 0)),
            "wrong_count": int(row_dict.get("wrong_count", 0)),
            "last_studied": str(row_dict.get("last_studied", "")),
        }).eq("id", int(row_dict["id"])) .execute()
        return
    # CSV mode: update the persisted CSV file with only the changed stats to avoid overwriting unintended session state
    df = load_words()
    df = normalize_df(df)
    mask = df["id"] == int(row_dict["id"])
    if mask.any():
        df.loc[mask, "correct_count"] = int(row_dict.get("correct_count", 0))
        df.loc[mask, "wrong_count"] = int(row_dict.get("wrong_count", 0))
        df.loc[mask, "last_studied"] = str(row_dict.get("last_studied", ""))
    else:
        # If the row isn't present for some reason, append it.
        append_df = normalize_df(pd.DataFrame([row_dict]))
        df = pd.concat([df, append_df], ignore_index=True)
    save_words(df)


def set_words(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_df(df)
    st.session_state[SESSION_WORDS_KEY] = df
    return df


def words_for_session(force_reload: bool = False) -> pd.DataFrame:
    if force_reload or SESSION_WORDS_KEY not in st.session_state:
        return set_words(load_words())
    return normalize_df(st.session_state[SESSION_WORDS_KEY])


def load_words() -> pd.DataFrame:
    if supabase_enabled():
        rows = supabase_client().table(SUPABASE_TABLE).select("*").order("id").execute().data or []
        if rows:
            return normalize_df(pd.DataFrame(rows))
        df = pd.DataFrame(SAMPLE_WORDS, columns=COLUMNS)
        save_words(df)
        return df
    if not DATA_FILE.exists():
        save_words(pd.DataFrame(SAMPLE_WORDS, columns=COLUMNS))
    return normalize_df(pd.read_csv(DATA_FILE, keep_default_na=False))


def esc(value: object) -> str:
    return html.escape(str(value))


def answer_diff_html(expected: object, actual: object) -> str:
    expected_text = str(expected).strip()
    actual_text = str(actual).strip()
    if not actual_text:
        return '<span class="diff-missing">未入力</span>'

    pieces: list[str] = []
    matcher = SequenceMatcher(None, actual_text.lower(), expected_text.lower())
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        actual_part = esc(actual_text[i1:i2])
        expected_part = esc(expected_text[j1:j2])
        if tag == "equal":
            pieces.append(actual_part)
        elif tag == "replace":
            pieces.append(f'<span class="diff-wrong">{actual_part}</span>')
        elif tag == "delete":
            pieces.append(f'<span class="diff-extra">{actual_part}</span>')
        elif tag == "insert":
            pieces.append(f'<span class="diff-missing">[{expected_part}]</span>')
    return "".join(pieces)


def render_pronunciation_button(word: object) -> None:
    speak_text = str(word).strip()
    if not speak_text:
        return

    components.html(
        f"""
        <button id="speak-word" type="button" aria-label="発音を再生">
          <span class="speaker-icon">▶</span>
          <span>発音</span>
        </button>
        <script>
          const button = document.getElementById("speak-word");
          const text = {json.dumps(speak_text)};
          button.addEventListener("click", () => {{
            if (!("speechSynthesis" in window)) {{
              button.textContent = "このブラウザでは音声再生できません";
              return;
            }}
            window.speechSynthesis.cancel();
            const utterance = new SpeechSynthesisUtterance(text);
            utterance.lang = "en-US";
            utterance.rate = 0.88;
            utterance.pitch = 1;
            window.speechSynthesis.speak(utterance);
          }});
        </script>
        <style>
          html, body {{
            margin: 0;
            padding: 0;
            background: transparent;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
          }}
          #speak-word {{
            align-items: center;
            background: #ffffff;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            color: #172033;
            display: inline-flex;
            font-size: 16px;
            font-weight: 800;
            gap: 0.45rem;
            justify-content: center;
            min-height: 46px;
            padding: 0 0.95rem;
            width: 100%;
          }}
          #speak-word:active {{
            background: #eef4ff;
            border-color: #2f6fed;
          }}
          .speaker-icon {{
            color: #2f6fed;
            font-size: 0.9rem;
          }}
        </style>
        """,
        height=54,
    )


def enable_return_to_next() -> None:
    components.html(
        """
        <script>
          const parentDoc = window.parent.document;
          if (parentDoc.__vocabReturnToNextHandler) {
            parentDoc.removeEventListener("keydown", parentDoc.__vocabReturnToNextHandler);
          }
          parentDoc.__vocabReturnToNextHandler = (event) => {
            if (event.key !== "Enter" || event.shiftKey || event.metaKey || event.ctrlKey || event.altKey || event.isComposing) {
              return;
            }
            const active = parentDoc.activeElement;
            if (active && active.tagName === "TEXTAREA") {
              return;
            }
            const nextButton = Array.from(parentDoc.querySelectorAll("button"))
              .find((button) => button.innerText.trim() === "次へ" && !button.disabled);
            if (nextButton) {
              event.preventDefault();
              nextButton.click();
            }
          };
          parentDoc.addEventListener("keydown", parentDoc.__vocabReturnToNextHandler);
        </script>
        """,
        height=0,
    )


def focus_answer_input() -> None:
    components.html(
        """
        <script>
          const parentDoc = window.parent.document;
          const focusAnswerInput = () => {
            const inputs = Array.from(parentDoc.querySelectorAll("input"));
            const answerInput = inputs.find((input) =>
              input.getAttribute("aria-label") === "英単語を入力" && !input.disabled
            );
            if (answerInput) {
              answerInput.focus({ preventScroll: true });
            }
          };
          setTimeout(focusAnswerInput, 80);
          setTimeout(focusAnswerInput, 300);
        </script>
        """,
        height=0,
    )


def today() -> str:
    return date.today().isoformat()


def norm(value: str) -> str:
    return value.strip().lower()


def with_scores(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_correct"] = pd.to_numeric(work["correct_count"], errors="coerce").fillna(0).astype(int)
    work["_wrong"] = pd.to_numeric(work["wrong_count"], errors="coerce").fillna(0).astype(int)
    work["weakness_score"] = work["_wrong"] - work["_correct"]
    return work


def priority(df: pd.DataFrame) -> pd.DataFrame:
    work = with_scores(df)
    work["_last"] = pd.to_datetime(work["last_studied"], errors="coerce").fillna(pd.Timestamp("1970-01-01"))
    return work.sort_values(["weakness_score", "_last", "_wrong", "word"], ascending=[False, True, False, True]).drop(columns=["_correct", "_wrong", "_last"])


def newest_first(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_id_sort"] = pd.to_numeric(work["id"], errors="coerce").fillna(0)
    return work.sort_values("_id_sort", ascending=False).drop(columns=["_id_sort"])


def ids_by_frequency(df: pd.DataFrame, ordered: pd.DataFrame) -> tuple[list[int], list[int]]:
    if ordered.empty:
        return [], []
    flags = df.set_index("id")["low_frequency"].apply(normalize_bool).to_dict()
    normal: list[int] = []
    reduced: list[int] = []
    for value in ordered["id"].tolist():
        word_id = int(value)
        if flags.get(word_id, False):
            reduced.append(word_id)
        else:
            normal.append(word_id)
    return normal, reduced


def mixed_ids(df: pd.DataFrame) -> list[int]:
    if df.empty:
        return []
    scored = with_scores(df)
    new_mask = (scored["_correct"] + scored["_wrong"]) == 0
    difficult_mask = scored["weakness_score"] > 0
    new_normal, new_reduced = ids_by_frequency(scored, newest_first(scored[new_mask]))
    difficult_normal, difficult_reduced = ids_by_frequency(scored, priority(scored[~new_mask & difficult_mask]))
    regular_normal, regular_reduced = ids_by_frequency(scored, priority(scored[~new_mask & ~difficult_mask]))
    return new_normal + difficult_normal + regular_normal + new_reduced + difficult_reduced + regular_reduced


def next_id(df: pd.DataFrame, current: int | None = None, recent_ids: list[int] | None = None) -> int | None:
    ids = mixed_ids(df)
    if not ids:
        return None
    if current not in ids or len(ids) == 1:
        candidates = ids
    else:
        start = ids.index(current) + 1
        candidates = ids[start:] + ids[:start]
    recent = set(int(value) for value in (recent_ids or [])[-LOW_FREQUENCY_GAP:])
    low_frequency = df.set_index("id")["low_frequency"].apply(normalize_bool).to_dict()
    recent_has_low_frequency = any(low_frequency.get(word_id, False) for word_id in recent)
    has_normal_alternative = any(
        candidate != current and not low_frequency.get(candidate, False)
        for candidate in candidates
    )
    for candidate in candidates:
        if len(ids) > 1 and candidate == current:
            continue
        if low_frequency.get(candidate, False) and has_normal_alternative and recent_has_low_frequency:
            continue
        return candidate
    return candidates[0] if candidates else ids[0]


def next_id_for_session(df: pd.DataFrame, current: int | None, recent_key: str) -> int | None:
    recent = [
        int(value)
        for value in st.session_state.get(recent_key, [])
        if row_by_id(df, int(value)) is not None
    ]
    if current is not None:
        recent.append(int(current))
    recent = recent[-LOW_FREQUENCY_GAP:]
    st.session_state[recent_key] = recent
    return next_id(df, current, recent)


def is_first_quiz_attempt(result: object, word_id: int) -> bool:
    return not isinstance(result, dict) or int(result.get("id", -1)) != int(word_id)


def pushed_history(history: list[int], word_id: int, limit: int = 30) -> list[int]:
    updated: list[int] = []
    for value in history:
        normalized_value = int(value)
        if normalized_value != int(word_id) and normalized_value not in updated:
            updated.append(normalized_value)
    updated.append(int(word_id))
    return updated[-limit:]


def pop_previous_id(history: list[int], df: pd.DataFrame) -> tuple[int | None, list[int]]:
    remaining = [int(value) for value in history]
    while remaining:
        previous = remaining.pop()
        if row_by_id(df, previous) is not None:
            return previous, remaining
    return None, []


def row_by_id(df: pd.DataFrame, word_id: int):
    rows = df[df["id"] == word_id]
    return None if rows.empty else rows.iloc[0]


def update_stats(df: pd.DataFrame, word_id: int, correct: bool) -> pd.DataFrame:
    df = df.copy()
    mask = df["id"] == word_id
    if not mask.any():
        return df
    col = "correct_count" if correct else "wrong_count"
    df.loc[mask, col] = df.loc[mask, col].astype(int) + 1
    df.loc[mask, "last_studied"] = today()
    df = normalize_df(df)
    row = df[df["id"] == word_id].iloc[0]
    if supabase_enabled():
        save_stats(row)
    else:
        save_words(df)
    return set_words(df)


def update_low_frequency(df: pd.DataFrame, word_id: int, low_frequency: bool) -> pd.DataFrame:
    df = df.copy()
    mask = df["id"] == word_id
    if not mask.any():
        return df
    df.loc[mask, "low_frequency"] = bool(low_frequency)
    df = normalize_df(df)
    if supabase_enabled():
        try:
            supabase_client().table(SUPABASE_TABLE).update({"low_frequency": bool(low_frequency)}).eq("id", int(word_id)).execute()
        except Exception as exc:
            raise LowFrequencySaveError("Supabaseに low_frequency 列がまだありません。Supabase SQL Editorで supabase_schema.sql を実行してから、アプリを再読み込みしてください。") from exc
    else:
        save_words(df)
    return set_words(df)


def upsert_word(df: pd.DataFrame, values: dict[str, object]) -> tuple[pd.DataFrame, bool]:
    df = df.copy()
    normalized_word = norm(values["word"])
    mask = df["word"].astype(str).str.strip().str.lower() == normalized_word
    if mask.any():
        idx = df[mask].index[0]
        for key, value in values.items():
            df.at[idx, key] = value
        created = False
    else:
        values = {**values, "id": int(df["id"].max()) + 1 if not df.empty else 1, "correct_count": 0, "wrong_count": 0, "last_studied": ""}
        df = pd.concat([df, pd.DataFrame([values])], ignore_index=True)
        created = True
    df = normalize_df(df)
    save_row = df[df["word"].astype(str).str.strip().str.lower() == normalized_word].iloc[0].to_dict()
    if supabase_enabled():
        save_rows([save_row])
    else:
        save_words(df)
    return set_words(df), created


def word_forms_for_blank(word: object) -> list[str]:
    base = str(word).strip().lower()
    if not base:
        return []
    forms = {base, f"{base}s", f"{base}ed", f"{base}ing"}
    if base.endswith("e") and len(base) > 1:
        stem = base[:-1]
        forms.update({f"{base}d", f"{stem}ing"})
    if base.endswith("y") and len(base) > 1:
        stem = base[:-1]
        forms.update({f"{stem}ies", f"{stem}ied"})
    return sorted(forms, key=len, reverse=True)


def blank_sentence(example: str, word: str) -> str:
    forms = word_forms_for_blank(word)
    if not forms:
        return example
    pattern = re.compile(rf"\b(?:{'|'.join(re.escape(form) for form in forms)})\b", re.IGNORECASE)
    return pattern.sub("_____", example, count=1) if pattern.search(example) else f"_____ {example}"


def render_card(row, show_answer: bool = True) -> None:
    if show_answer:
        answer_html = f"""
      <div class="meaning">{esc(row['meaning_ja'])}</div>
      <div class="example-ja">{esc(row['example_ja'])}</div>
        """
    else:
        answer_html = '<div class="answer-placeholder">日本語訳はまだ隠れています。</div>'
    frequency_pill = '<span class="pill">頻度低</span>' if normalize_bool(row.get("low_frequency", False)) else ""
    st.markdown(f"""
    <div class="word-card">
      <div><span class="pill">{esc(row['category'])}</span><span class="pill">{esc(row['part_of_speech'])}</span><span class="pill">Lv {esc(row['difficulty'])}</span>{frequency_pill}</div>
      <div class="word-title">{esc(row['word'])}</div>
      <div class="pronunciation">{esc(row['pronunciation'])}</div>
      <div class="example-en">{esc(row['example_en'])}</div>
      {answer_html}
      <div class="stats-line">正解 {int(row['correct_count'])} ・ 不正解 {int(row['wrong_count'])} ・ 最終 {esc(row['last_studied'] or '-')}</div>
    </div>
    """, unsafe_allow_html=True)
    render_pronunciation_button(row["word"])


def register_screen(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("単語登録")
    with st.form("word_form", clear_on_submit=True):
        word = st.text_input("英単語")
        pronunciation = st.text_input("発音メモ")
        part_of_speech = st.selectbox("品詞", PARTS_OF_SPEECH, index=1)
        meaning = st.text_area("日本語の意味", height=80)
        example_en = st.text_area("英語の例文", height=90)
        example_ja = st.text_area("例文の日本語訳", height=90)
        category = st.text_input("カテゴリ", value="Uncategorized")
        difficulty = st.selectbox("難易度", ["1", "2", "3", "4", "5"], index=2)
        low_frequency = st.checkbox("この単語の出題頻度を下げる")
        submitted = st.form_submit_button("保存")
    if submitted:
        if not word.strip() or not meaning.strip():
            st.error("英単語と日本語の意味は必須です。")
        else:
            df, created = upsert_word(df, {"word": word.strip(), "pronunciation": pronunciation.strip(), "part_of_speech": part_of_speech, "meaning_ja": meaning.strip(), "example_en": example_en.strip(), "example_ja": example_ja.strip(), "category": category.strip() or "Uncategorized", "difficulty": difficulty, "low_frequency": low_frequency})
            st.success("新しい単語を登録しました。" if created else "既存の単語を更新しました。")
    with st.expander("登録済み単語"):
        st.dataframe(df[["word", "part_of_speech", "meaning_ja", "category", "difficulty", "low_frequency", "correct_count", "wrong_count", "last_studied"]].rename(columns={"word": "英単語", "part_of_speech": "品詞", "meaning_ja": "意味", "category": "カテゴリ", "difficulty": "難易度", "low_frequency": "頻度低", "correct_count": "正解", "wrong_count": "不正解", "last_studied": "最終学習日"}), use_container_width=True, hide_index=True)
    return df


def study_screen(df: pd.DataFrame) -> pd.DataFrame:
    key = "study_current_id"
    reveal_key = "study_answer_visible"
    viewed_key = "study_viewed_id"
    recent_key = "study_recent_ids"
    history_key = "study_history_ids"
    if key not in st.session_state or row_by_id(df, st.session_state[key]) is None:
        st.session_state[key] = next_id_for_session(df, None, recent_key)
        st.session_state[reveal_key] = False
    row = row_by_id(df, st.session_state[key])
    if row is None:
        st.info("単語がありません。")
        return df
    if st.session_state.get(viewed_key) != int(row["id"]):
        st.session_state[viewed_key] = int(row["id"])
        st.session_state[reveal_key] = False
    show_answer = bool(st.session_state.get(reveal_key, False))
    st.subheader("学習カード")
    st.caption("新しい単語を先に出し、その後に苦手数（不正解 - 正解）が高い単語を出します。頻度低の単語は直近20回に出ている間、通常単語を優先します。")
    render_card(row, show_answer=show_answer)
    current_low_frequency = normalize_bool(row.get("low_frequency", False))
    low_frequency = st.checkbox(
        "この単語の出題頻度を下げる",
        value=current_low_frequency,
        key=f"study_low_frequency_{int(row['id'])}",
    )
    if low_frequency != current_low_frequency:
        try:
            df = update_low_frequency(df, int(row["id"]), low_frequency)
        except LowFrequencySaveError as exc:
            st.error(str(exc))
            st.caption("保存はまだ完了していません。SQL実行後にもう一度チェックしてください。")
        else:
            st.toast("出題頻度の設定を保存しました。")
            st.rerun()
    if not show_answer:
        if st.button("表示", type="primary", use_container_width=True):
            st.session_state[reveal_key] = True
            st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("覚えた", type="primary", use_container_width=True):
        df = update_stats(df, int(row["id"]), True)
        st.session_state[history_key] = pushed_history(st.session_state.get(history_key, []), int(row["id"]))
        st.session_state[key] = next_id_for_session(df, int(row["id"]), recent_key)
        st.session_state[reveal_key] = False
        st.rerun()
    if c2.button("苦手", use_container_width=True):
        df = update_stats(df, int(row["id"]), False)
        st.session_state[history_key] = pushed_history(st.session_state.get(history_key, []), int(row["id"]))
        st.session_state[key] = next_id_for_session(df, int(row["id"]), recent_key)
        st.session_state[reveal_key] = False
        st.rerun()
    history = st.session_state.get(history_key, [])
    if st.button("次へ", use_container_width=True):
        st.session_state[history_key] = pushed_history(history, int(row["id"]))
        st.session_state[key] = next_id_for_session(df, int(row["id"]), recent_key)
        st.session_state[reveal_key] = False
        st.rerun()
    if st.button("前へ", use_container_width=True, disabled=not bool(history)):
        previous_id, remaining_history = pop_previous_id(history, df)
        st.session_state[history_key] = remaining_history
        if previous_id is not None:
            st.session_state[key] = previous_id
        st.session_state[reveal_key] = False
        st.rerun()
    return df


def quiz_screen(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    available = df if mode == "written" else df[df["example_en"].astype(str).str.strip() != ""]
    if available.empty:
        st.info("問題に使える単語がありません。")
        return df
    current_key = f"{mode}_current_id"
    result_key = f"{mode}_result"
    answer_key = f"{mode}_answer"
    recent_key = f"{mode}_recent_ids"
    history_key = f"{mode}_history_ids"
    if current_key not in st.session_state or row_by_id(available, st.session_state[current_key]) is None:
        st.session_state[current_key] = next_id_for_session(available, None, recent_key)
    row = row_by_id(available, st.session_state[current_key])
    prompt = row["meaning_ja"] if mode == "written" else blank_sentence(row["example_en"], row["word"])
    hint = f"{row['part_of_speech']} ・ {row['category']} ・ Lv {row['difficulty']}" if mode == "written" else row["example_ja"]
    st.subheader("筆記問題" if mode == "written" else "穴埋め問題")
    st.caption("新しい単語を先に出し、1回目で正解が多い単語は後半へ、頻度低の単語は直近20回に出ている間は通常単語を優先します。")
    st.markdown(f'<div class="quiz-card"><div class="quiz-label">問題</div><div class="quiz-prompt">{esc(prompt)}</div><div class="hint-line">{esc(hint)}</div></div>', unsafe_allow_html=True)
    current_low_frequency = normalize_bool(row.get("low_frequency", False))
    low_frequency = st.checkbox(
        "この単語の出題頻度を下げる",
        value=current_low_frequency,
        key=f"{mode}_low_frequency_{int(row['id'])}",
    )
    if low_frequency != current_low_frequency:
        try:
            df = update_low_frequency(df, int(row["id"]), low_frequency)
        except LowFrequencySaveError as exc:
            st.error(str(exc))
            st.caption("保存はまだ完了していません。SQL実行後にもう一度チェックしてください。")
        else:
            st.toast("出題頻度の設定を保存しました。")
            st.rerun()
    result = st.session_state.get(result_key)
    if result and result["id"] == int(row["id"]):
        (st.success if result["correct"] else st.error)(f"{'正解' if result['correct'] else '不正解'}です。正解: {result['expected']}")
        if not result["correct"]:
            st.markdown(
                f"""
                <div class="answer-review">
                  <div class="answer-review-label">あなたの回答</div>
                  <div class="answer-review-text">{answer_diff_html(result["expected"], result.get("answer", ""))}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        render_pronunciation_button(result["expected"])
        if result["correct"]:
            enable_return_to_next()
            st.caption("次へボタン、またはReturnキーで次の問題へ進めます。")
            history = st.session_state.get(history_key, [])
            if st.button("次へ", type="primary", use_container_width=True):
                st.session_state.pop(result_key, None)
                st.session_state[answer_key] = ""
                st.session_state[history_key] = pushed_history(history, int(row["id"]))
                st.session_state[current_key] = next_id_for_session(available, int(row["id"]), recent_key)
                st.rerun()
            if st.button("前へ", use_container_width=True, disabled=not bool(history)):
                previous_id, remaining_history = pop_previous_id(history, available)
                st.session_state.pop(result_key, None)
                st.session_state[answer_key] = ""
                st.session_state[history_key] = remaining_history
                if previous_id is not None:
                    st.session_state[current_key] = previous_id
                st.rerun()
            return df
        st.caption("もう一度入力して、正解できたら次へ進めます。")
    with st.form(f"{mode}_form", clear_on_submit=True):
        answer = st.text_input("英単語を入力", key=answer_key)
        submitted = st.form_submit_button("判定")
    focus_answer_input()
    if submitted:
        correct = norm(answer) == norm(row["word"])
        if is_first_quiz_attempt(st.session_state.get(result_key), int(row["id"])):
            df = update_stats(df, int(row["id"]), correct)
        st.session_state[result_key] = {"id": int(row["id"]), "correct": correct, "expected": row["word"], "answer": answer}
        st.rerun()
    return df


def review_screen(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("復習")
    p = priority(df)
    render_card(p.iloc[0])
    c1, c2 = st.columns(2)
    c1.metric("単語数", len(df))
    c2.metric("不正解合計", int(df["wrong_count"].sum()))
    st.caption("苦手数（不正解 - 正解）が多い順に並びます。")
    st.dataframe(p[["word", "part_of_speech", "meaning_ja", "weakness_score", "correct_count", "wrong_count", "last_studied", "category", "difficulty", "low_frequency"]].rename(columns={"word": "英単語", "part_of_speech": "品詞", "meaning_ja": "意味", "weakness_score": "苦手数", "correct_count": "正解", "wrong_count": "不正解", "last_studied": "最終学習日", "category": "カテゴリ", "difficulty": "難易度", "low_frequency": "頻度低"}), use_container_width=True, hide_index=True)
    return df


def last_ai_date() -> str:
    if supabase_enabled():
        rows = supabase_client().table(SUPABASE_SETTINGS_TABLE).select("value").eq("key", "last_ai_words_date").limit(1).execute().data or []
        return rows[0]["value"] if rows else ""
    marker = Path(".last_ai_words_date")
    return marker.read_text().strip() if marker.exists() else ""


def set_last_ai_date() -> None:
    if supabase_enabled():
        supabase_client().table(SUPABASE_SETTINGS_TABLE).upsert({"key": "last_ai_words_date", "value": today()}, on_conflict="key").execute()
    else:
        Path(".last_ai_words_date").write_text(today())


def saved_ai_category() -> str:
    if supabase_enabled():
        rows = supabase_client().table(SUPABASE_SETTINGS_TABLE).select("value").eq("key", "ai_category_hint").limit(1).execute().data or []
        return rows[0]["value"] if rows else ""
    marker = Path(".ai_category_hint")
    return marker.read_text().strip() if marker.exists() else ""


def set_saved_ai_category(category: str) -> None:
    value = category.strip()
    if supabase_enabled():
        supabase_client().table(SUPABASE_SETTINGS_TABLE).upsert({"key": "ai_category_hint", "value": value}, on_conflict="key").execute()
    else:
        Path(".ai_category_hint").write_text(value)


def generate_ai_words(df: pd.DataFrame, count: int, category: str, difficulty: str, model: str) -> tuple[pd.DataFrame, list[str]]:
    api_key = config("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")
    from openai import OpenAI
    from pydantic import BaseModel, Field

    class AiWord(BaseModel):
        word: str
        pronunciation: str
        part_of_speech: str = Field(description="Part of speech: noun, verb, adjective, adverb, phrase, or other")
        meaning_ja: str
        example_en: str
        example_ja: str
        category: str
        difficulty: str = Field(pattern="^[1-5]$")

    class Batch(BaseModel):
        words: list[AiWord]

    existing = ", ".join(df["word"].astype(str).tolist()[:500])
    prompt = f"Generate {count} useful English vocabulary entries for a Japanese learner. Do not include these words: {existing}. Include part_of_speech as noun, verb, adjective, adverb, phrase, or other. Category hint: {category or 'any practical category'}. Difficulty hint: {difficulty}. Return Japanese meanings and translations."
    parsed = OpenAI(api_key=api_key).responses.parse(model=model or DEFAULT_AI_MODEL, instructions="You are an English vocabulary coach for Japanese learners.", input=prompt, text_format=Batch).output_parsed
    existing_set = set(df["word"].astype(str).str.lower().str.strip())
    rows = []
    added = []
    next_word_id = int(df["id"].max()) + 1 if not df.empty else 1
    for item in parsed.words:
        values = item.model_dump()
        word = values["word"].strip()
        if not word or word.lower() in existing_set:
            continue
        values["part_of_speech"] = normalize_pos(values.get("part_of_speech"))
        rows.append({"id": next_word_id, **values, "correct_count": 0, "wrong_count": 0, "last_studied": ""})
        added.append(word)
        existing_set.add(word.lower())
        next_word_id += 1
        if len(rows) >= count:
            break
    if rows:
        df = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
        if supabase_enabled():
            save_rows(rows)
        else:
            save_words(df)
        set_last_ai_date()
    return set_words(df), added


def classify_existing_pos(df: pd.DataFrame, model: str, api_key: str) -> tuple[pd.DataFrame, int]:
    missing = df[df["part_of_speech"].apply(normalize_pos) == "other"]
    if missing.empty:
        return df, 0
    from openai import OpenAI
    from pydantic import BaseModel, Field

    class PosItem(BaseModel):
        word: str
        part_of_speech: str = Field(pattern="^(noun|verb|adjective|adverb|phrase|other)$")

    class PosBatch(BaseModel):
        words: list[PosItem]

    lines = "\n".join(f"- {r.word}: {r.meaning_ja}; example: {r.example_en}" for r in missing.itertuples())
    parsed = OpenAI(api_key=api_key).responses.parse(
        model=model or DEFAULT_AI_MODEL,
        instructions="Classify each English word using only noun, verb, adjective, adverb, phrase, or other.",
        input=f"Classify these vocabulary entries and return one result for each:\n{lines}",
        text_format=PosBatch,
    ).output_parsed
    lookup = {item.word.strip().lower(): normalize_pos(item.part_of_speech) for item in parsed.words}
    updated = normalize_df(df)
    changed = []
    for idx, row in missing.iterrows():
        part = lookup.get(str(row["word"]).strip().lower(), "other")
        if part != "other":
            updated.at[idx, "part_of_speech"] = part
            changed.append(idx)
    if not changed:
        return set_words(updated), 0
    updated = normalize_df(updated)
    if supabase_enabled():
        save_rows(updated.loc[changed, COLUMNS].to_dict("records"))
    else:
        save_words(updated)
    return set_words(updated), len(changed)


def ai_screen(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("AIで単語追加")
    api_key = config("OPENAI_API_KEY")
    model_default = config("OPENAI_MODEL", DEFAULT_AI_MODEL)
    if "ai_category_hint" not in st.session_state:
        st.session_state["ai_category_hint"] = saved_ai_category()
    st.info(f"最終AI追加日: {last_ai_date() or '-'} / 現在の単語数: {len(df)}")
    if not api_key:
        st.warning("OPENAI_API_KEY をSecretsに設定すると使えます。")
    with st.form("ai_form"):
        count = st.number_input("追加する単語数", 1, 20, 5)
        category = st.text_input("カテゴリの希望", placeholder="Business, Academic など", key="ai_category_hint")
        difficulty = st.selectbox("難易度", ["3から5を中心にする", "1から2の基礎", "3の中級", "4から5の上級"], index=3)
        model = st.text_input("モデル", value=model_default)
        force = st.checkbox("今日すでに追加済みでも実行する")
        submitted = st.form_submit_button("AI追加", disabled=not bool(api_key))
    if submitted:
        set_saved_ai_category(category)
        if last_ai_date() == today() and not force:
            st.info("今日はすでに追加済みです。")
            return df
        try:
            with st.spinner("AIが単語を作成しています..."):
                df, added = generate_ai_words(df, int(count), category, difficulty, model)
            st.success(f"{len(added)}語を追加しました: {', '.join(added)}" if added else "新しい単語は追加されませんでした。")
        except Exception as exc:
            st.error(f"AI生成に失敗しました: {exc}")
    missing_pos = int((df["part_of_speech"].apply(normalize_pos) == "other").sum())
    with st.expander("既存単語の品詞を補完"):
        st.write(f"品詞が未設定の単語: {missing_pos}語")
        if st.button("品詞補完", type="primary", use_container_width=True, disabled=not bool(api_key) or missing_pos == 0):
            try:
                with st.spinner("AIが既存単語の品詞を判定しています..."):
                    df, changed = classify_existing_pos(df, model_default, api_key)
            except Exception as exc:
                st.error(f"品詞の補完に失敗しました: {exc}")
            else:
                st.success(f"{changed}語の品詞を更新しました。")
                st.rerun()
    return df


def require_password() -> bool:
    password = config("APP_PASSWORD")
    if not password or st.session_state.get("authenticated"):
        return True
    st.title("英単語帳")
    with st.form("login"):
        entered = st.text_input("パスワード", type="password")
        submitted = st.form_submit_button("開く")
    if submitted:
        if entered == password:
            st.session_state["authenticated"] = True
            st.rerun()
        st.error("パスワードが違います。")
    return False


def css() -> None:
    st.markdown("""
    <style>
      .stApp { background:#f5f7fb; color:#172033; }
      .block-container { max-width:720px; padding:1rem .85rem 5rem; }
      h1 { font-size:1.65rem!important; line-height:1.2!important; }
      div[data-testid="stRadio"] > div { display:flex; flex-wrap:wrap; gap:.35rem; }
      div[data-testid="stRadio"] label { background:#fff; border:1px solid #dce3ee; border-radius:8px; color:#172033; padding:.35rem .6rem; min-height:42px; }
      div[data-testid="stRadio"] label * { color:#172033!important; }
      .word-card,.quiz-card { background:#fff; border:1px solid #e1e7f0; border-radius:8px; padding:1rem; box-shadow:0 8px 24px rgba(24,39,75,.08); margin:.4rem 0 1rem; }
      .pill { background:#eef6f1; color:#24533b; border:1px solid #d7ebdf; border-radius:999px; display:inline-flex; align-items:center; min-height:28px; padding:0 .65rem; font-size:.8rem; font-weight:700; margin-right:.4rem; }
      .word-title { color:#111827; font-size:2.2rem; font-weight:800; line-height:1.05; overflow-wrap:anywhere; margin-top:.8rem; }
      .pronunciation,.example-ja,.stats-line,.hint-line { color:#596579; font-size:.95rem; line-height:1.55; margin-top:.5rem; overflow-wrap:anywhere; }
      .meaning { color:#182033; font-size:1.15rem; font-weight:700; margin-top:1rem; overflow-wrap:anywhere; }
      .example-en { background:#f7f9fc; border-left:4px solid #2f6fed; color:#1f2937; margin-top:1rem; padding:.8rem; line-height:1.55; overflow-wrap:anywhere; }
      .answer-placeholder { background:#f7f9fc; border:1px dashed #cbd5e1; border-radius:8px; color:#687385; font-size:.95rem; margin-top:.85rem; padding:.75rem; text-align:center; }
      .answer-review { background:#fff7ed; border:1px solid #fed7aa; border-radius:8px; margin:.8rem 0 1rem; padding:.8rem; }
      .answer-review-label { color:#9a3412; font-size:.82rem; font-weight:800; margin-bottom:.35rem; }
      .answer-review-text { color:#111827; font-size:1.15rem; font-weight:800; letter-spacing:0; overflow-wrap:anywhere; }
      .diff-wrong { background:#fee2e2; border-bottom:3px solid #ef4444; border-radius:4px; color:#991b1b; padding:0 .08rem; }
      .diff-extra { background:#fef3c7; border-bottom:3px solid #f59e0b; border-radius:4px; color:#92400e; padding:0 .08rem; }
      .diff-missing { background:#dbeafe; border-bottom:3px solid #2f6fed; border-radius:4px; color:#1d4ed8; padding:0 .16rem; }
      .quiz-label { color:#596579; font-size:0.82rem; font-weight:700; }
      .quiz-prompt { color:#111827; font-size:1.25rem; font-weight:800; line-height:1.35; overflow-wrap:anywhere; }
      input, textarea, select { background:#fff!important; color:#172033!important; -webkit-text-fill-color:#172033!important; caret-color:#172033!important; border-color:#cbd5e1!important; font-size:16px!important; }
      input::placeholder, textarea::placeholder { color:#8a94a6!important; -webkit-text-fill-color:#8a94a6!important; opacity:1!important; }
      label, label *, div[data-testid="stWidgetLabel"], div[data-testid="stWidgetLabel"] * { color:#172033!important; }
      div.stButton > button, div[data-testid="stFormSubmitButton"] button { background:#fff!important; border:1px solid #cbd5e1!important; border-radius:8px; color:#172033!important; min-height:46px; font-weight:800; width:100%; }
      button[data-testid="stBaseButton-primary"], button[data-testid="stBaseButton-primaryFormSubmit"] { background:#2f6fed!important; border-color:#2f6fed!important; color:#fff!important; }
      button * { color:inherit!important; }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    css()
    if not require_password():
        return
    st.title("英単語帳")
    df = words_for_session()

    menu = st.sidebar.radio("モード", ["学習カード", "筆記問題", "穴埋め問題", "復習", "単語登録", "AI追加"], index=0)
    st.sidebar.write(f"単語数: {len(df)}")

    if menu == "学習カード":
        df = study_screen(df)
    elif menu == "筆記問題":
        df = quiz_screen(df, "written")
    elif menu == "穴埋め問題":
        df = quiz_screen(df, "fill")
    elif menu == "復習":
        df = review_screen(df)
    elif menu == "単語登録":
        df = register_screen(df)
    elif menu == "AI追加":
        df = ai_screen(df)

    # show storage info
    if supabase_enabled():
        st.caption("保存先: Supabase")
    else:
        st.caption(f"保存先: {DATA_FILE}")


if __name__ == "__main__":
    main()
