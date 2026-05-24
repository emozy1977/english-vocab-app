from __future__ import annotations

import html
import os
import re
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_FILE = Path(__file__).with_name("words.csv")
COLUMNS = ["id", "word", "pronunciation", "part_of_speech", "meaning_ja", "example_en", "example_ja", "category", "difficulty", "correct_count", "wrong_count", "last_studied"]
COUNT_COLUMNS = ["correct_count", "wrong_count"]
SUPABASE_TABLE = "words"
SUPABASE_SETTINGS_TABLE = "app_settings"
DEFAULT_AI_MODEL = "gpt-5.4-mini"
SESSION_WORDS_KEY = "words_df"

SAMPLE_WORDS = [
    [1, "incorporate", "in-KOR-puh-rayt", "verb", "取り入れる、組み込む", "We need to incorporate user feedback into the next version.", "次のバージョンにユーザーの意見を取り入れる必要があります。", "Business", "4", 0, 0, ""],
    [2, "consolidate", "kun-SOL-ih-dayt", "verb", "統合する、強化する", "The team will consolidate multiple reports into one dashboard.", "チームは複数のレポートを1つのダッシュボードに統合します。", "Business", "4", 0, 0, ""],
    [3, "appropriate", "uh-PROH-pree-uht", "adjective", "適切な", "Please choose the most appropriate response for the situation.", "その状況に最も適切な返答を選んでください。", "Academic", "3", 0, 0, ""],
    [4, "implement", "IM-pluh-ment", "verb", "実行する、実装する", "The company plans to implement a new training program.", "会社は新しい研修プログラムを実施する予定です。", "Business", "3", 0, 0, ""],
    [5, "overlook", "oh-ver-LOOK", "verb", "見落とす、大目に見る", "It is easy to overlook small errors when you are tired.", "疲れていると小さな誤りを見落としやすいです。", "Daily", "3", 0, 0, ""],
    [6, "fatigue", "fuh-TEEG", "noun", "疲労", "Long meetings can cause mental fatigue.", "長い会議は精神的な疲労を引き起こすことがあります。", "Health", "2", 0, 0, ""],
    [7, "retention", "ree-TEN-shun", "noun", "保持、定着", "Regular review improves vocabulary retention.", "定期的な復習は語彙の定着を高めます。", "Learning", "4", 0, 0, ""],
    [8, "elaborate", "ih-LAB-uh-rayt", "verb", "詳しく説明する", "Could you elaborate on your main idea?", "主な考えについて詳しく説明してもらえますか。", "Academic", "3", 0, 0, ""],
    [9, "conversely", "KON-ver-slee", "adverb", "反対に、逆に", "Some tasks require speed; conversely, others require careful planning.", "速さが必要な作業もありますが、逆に慎重な計画が必要な作業もあります。", "Academic", "4", 0, 0, ""],
    [10, "recurrent", "ree-KUR-unt", "adjective", "繰り返し起こる", "The app helps users review recurrent mistakes.", "そのアプリはユーザーが繰り返し起こる間違いを復習するのに役立ちます。", "Academic", "4", 0, 0, ""],
]


def config(key: str, default: str = "") -> str:
    value = os.getenv(key, "").strip()
    if value:
        return value
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


def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = 0 if col in COUNT_COLUMNS else ""
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
    df["part_of_speech"] = df["part_of_speech"].astype(str).replace({"": "other"})
    return df


def save_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    normalized = normalize_df(pd.DataFrame(rows))
    if supabase_enabled():
        supabase_client().table(SUPABASE_TABLE).upsert(normalized[COLUMNS].to_dict("records"), on_conflict="word").execute()
        return
    current = load_words()
    merged = pd.concat([current, normalized], ignore_index=True)
    merged = merged.drop_duplicates(subset=["word"], keep="last")
    merged.to_csv(DATA_FILE, index=False)


def save_words(df: pd.DataFrame) -> None:
    df = normalize_df(df)
    if supabase_enabled():
        supabase_client().table(SUPABASE_TABLE).upsert(df[COLUMNS].to_dict("records"), on_conflict="word").execute()
        return
    df.to_csv(DATA_FILE, index=False)


def save_stats(row: pd.Series) -> None:
    if supabase_enabled():
        supabase_client().table(SUPABASE_TABLE).update({
            "correct_count": int(row["correct_count"]),
            "wrong_count": int(row["wrong_count"]),
            "last_studied": str(row["last_studied"]),
        }).eq("id", int(row["id"])).execute()
        return
    save_words(st.session_state.get(SESSION_WORDS_KEY, pd.DataFrame()))


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


def today() -> str:
    return date.today().isoformat()


def norm(value: str) -> str:
    return value.strip().lower()


def priority(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_last"] = pd.to_datetime(work["last_studied"], errors="coerce").fillna(pd.Timestamp("1970-01-01"))
    return work.sort_values(["wrong_count", "_last", "correct_count", "word"], ascending=[False, True, True, True]).drop(columns=["_last"])


def newest_first(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_id_sort"] = pd.to_numeric(work["id"], errors="coerce").fillna(0)
    return work.sort_values("_id_sort", ascending=False).drop(columns=["_id_sort"])


def mixed_ids(df: pd.DataFrame) -> list[int]:
    if df.empty:
        return []
    correct = pd.to_numeric(df["correct_count"], errors="coerce").fillna(0).astype(int)
    wrong = pd.to_numeric(df["wrong_count"], errors="coerce").fillna(0).astype(int)
    new_mask = (correct + wrong) == 0
    difficult_mask = wrong > 0
    buckets = {
        "difficult": [int(x) for x in priority(df[difficult_mask])["id"].tolist()],
        "new": [int(x) for x in newest_first(df[new_mask])["id"].tolist()],
        "regular": [int(x) for x in priority(df[~new_mask & ~difficult_mask])["id"].tolist()],
    }
    order = ["difficult", "new", "difficult", "regular"]
    ids: list[int] = []
    while any(buckets.values()):
        for name in order:
            if buckets[name]:
                ids.append(buckets[name].pop(0))
    return ids


def next_id(df: pd.DataFrame, current: int | None = None) -> int | None:
    ids = mixed_ids(df)
    if not ids:
        return None
    if current not in ids or len(ids) == 1:
        return ids[0]
    return ids[(ids.index(current) + 1) % len(ids)]


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


def upsert_word(df: pd.DataFrame, values: dict[str, str]) -> tuple[pd.DataFrame, bool]:
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


def blank_sentence(example: str, word: str) -> str:
    pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
    return pattern.sub("_____", example, count=1) if pattern.search(example) else f"_____ {example}"


def render_card(row, show_answer: bool = True) -> None:
    if show_answer:
        answer_html = f"""
      <div class="meaning">{esc(row['meaning_ja'])}</div>
      <div class="example-ja">{esc(row['example_ja'])}</div>
        """
    else:
        answer_html = '<div class="answer-placeholder">日本語訳はまだ隠れています。</div>'
    st.markdown(f"""
    <div class="word-card">
      <div><span class="pill">{esc(row['category'])}</span><span class="pill">{esc(row['part_of_speech'])}</span><span class="pill">Lv {esc(row['difficulty'])}</span></div>
      <div class="word-title">{esc(row['word'])}</div>
      <div class="pronunciation">{esc(row['pronunciation'])}</div>
      <div class="example-en">{esc(row['example_en'])}</div>
      {answer_html}
      <div class="stats-line">正解 {int(row['correct_count'])} ・ 不正解 {int(row['wrong_count'])} ・ 最終 {esc(row['last_studied'] or '-')}</div>
    </div>
    """, unsafe_allow_html=True)


def register_screen(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("単語登録")
    with st.form("word_form", clear_on_submit=True):
        word = st.text_input("英単語")
        pronunciation = st.text_input("発音メモ")
        part_of_speech = st.selectbox("品詞", ["noun", "verb", "adjective", "adverb", "phrase", "other"], index=1)
        meaning = st.text_area("日本語の意味", height=80)
        example_en = st.text_area("英語の例文", height=90)
        example_ja = st.text_area("例文の日本語訳", height=90)
        category = st.text_input("カテゴリ", value="Uncategorized")
        difficulty = st.selectbox("難易度", ["1", "2", "3", "4", "5"], index=2)
        submitted = st.form_submit_button("保存する")
    if submitted:
        if not word.strip() or not meaning.strip():
            st.error("英単語と日本語の意味は必須です。")
        else:
            df, created = upsert_word(df, {"word": word.strip(), "pronunciation": pronunciation.strip(), "part_of_speech": part_of_speech, "meaning_ja": meaning.strip(), "example_en": example_en.strip(), "example_ja": example_ja.strip(), "category": category.strip() or "Uncategorized", "difficulty": difficulty})
            st.success("新しい単語を登録しました。" if created else "既存の単語を更新しました。")
    with st.expander("登録済み単語"):
        st.dataframe(df[["word", "part_of_speech", "meaning_ja", "category", "difficulty", "correct_count", "wrong_count", "last_studied"]], width="stretch", hide_index=True)
    return df


def study_screen(df: pd.DataFrame) -> pd.DataFrame:
    key = "study_current_id"
    reveal_key = "study_answer_visible"
    viewed_key = "study_viewed_id"
    if key not in st.session_state or row_by_id(df, st.session_state[key]) is None:
        st.session_state[key] = next_id(df)
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
    st.caption("苦手な単語を優先しつつ、新しく追加した単語も混ぜて出します。")
    render_card(row, show_answer=show_answer)
    if not show_answer:
        if st.button("意味を表示", type="primary", width="stretch"):
            st.session_state[reveal_key] = True
            st.rerun()
    c1, c2 = st.columns(2)
    if c1.button("覚えた", type="primary", width="stretch"):
        df = update_stats(df, int(row["id"]), True)
        st.session_state[key] = next_id(df, int(row["id"]))
        st.session_state[reveal_key] = False
        st.rerun()
    if c2.button("苦手", width="stretch"):
        df = update_stats(df, int(row["id"]), False)
        st.session_state[key] = next_id(df, int(row["id"]))
        st.session_state[reveal_key] = False
        st.rerun()
    if st.button("次のカード", width="stretch"):
        st.session_state[key] = next_id(df, int(row["id"]))
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
    if current_key not in st.session_state or row_by_id(available, st.session_state[current_key]) is None:
        st.session_state[current_key] = next_id(available)
    row = row_by_id(available, st.session_state[current_key])
    prompt = row["meaning_ja"] if mode == "written" else blank_sentence(row["example_en"], row["word"])
    hint = f"{row['part_of_speech']} ・ {row['category']} ・ Lv {row['difficulty']}" if mode == "written" else row["example_ja"]
    st.subheader("筆記問題" if mode == "written" else "穴埋め問題")
    st.markdown(f'<div class="quiz-card"><div class="quiz-label">問題</div><div class="quiz-prompt">{esc(prompt)}</div><div class="hint-line">{esc(hint)}</div></div>', unsafe_allow_html=True)
    result = st.session_state.get(result_key)
    if result and result["id"] == int(row["id"]):
        (st.success if result["correct"] else st.error)(f"{'正解' if result['correct'] else '不正解'}です。正解: {result['expected']}")
        if st.button("次の問題", type="primary", width="stretch"):
            st.session_state.pop(result_key, None)
            st.session_state[current_key] = next_id(available, int(row["id"]))
            st.rerun()
        return df
    with st.form(f"{mode}_form"):
        answer = st.text_input("英単語を入力")
        submitted = st.form_submit_button("判定する")
    if submitted:
        correct = norm(answer) == norm(row["word"])
        df = update_stats(df, int(row["id"]), correct)
        st.session_state[result_key] = {"id": int(row["id"]), "correct": correct, "expected": row["word"]}
        st.rerun()
    return df


def review_screen(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("復習")
    p = priority(df)
    render_card(p.iloc[0])
    c1, c2 = st.columns(2)
    c1.metric("単語数", len(df))
    c2.metric("不正解合計", int(df["wrong_count"].sum()))
    st.dataframe(p[["word", "part_of_speech", "meaning_ja", "correct_count", "wrong_count", "last_studied", "category", "difficulty"]], width="stretch", hide_index=True)
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
        values["part_of_speech"] = str(values.get("part_of_speech") or "other").strip() or "other"
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


def ai_screen(df: pd.DataFrame) -> pd.DataFrame:
    st.subheader("AIで単語追加")
    st.info(f"最終AI追加日: {last_ai_date() or '-'} / 現在の単語数: {len(df)}")
    if not config("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY をSecretsに設定すると使えます。")
    with st.form("ai_form"):
        count = st.number_input("追加する単語数", 1, 20, 5)
        category = st.text_input("カテゴリの希望", placeholder="Business, Academic など")
        difficulty = st.selectbox("難易度", ["3から5を中心にする", "1から2の基礎", "3の中級", "4から5の上級"])
        model = st.text_input("モデル", value=config("OPENAI_MODEL", DEFAULT_AI_MODEL))
        force = st.checkbox("今日すでに追加済みでも実行する")
        submitted = st.form_submit_button("AIで今日分を追加", disabled=not bool(config("OPENAI_API_KEY")))
    if submitted:
        if last_ai_date() == today() and not force:
            st.info("今日はすでに追加済みです。")
            return df
        try:
            with st.spinner("AIが単語を作成しています..."):
                df, added = generate_ai_words(df, int(count), category, difficulty, model)
            st.success(f"{len(added)}語を追加しました: {', '.join(added)}" if added else "新しい単語は追加されませんでした。")
        except Exception as exc:
            st.error(f"AI生成に失敗しました: {exc}")
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
      .quiz-label { color:#596579; font-size:.82rem; font-weight:700; }
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
    st.set_page_config(page_title="英単語帳", page_icon="📚", layout="centered", initial_sidebar_state="collapsed")
    css()
    if not require_password():
        return
    df = words_for_session()
    st.title("英単語帳")
    st.caption(f"スマホのブラウザで使える英単語帳 ・ 保存先: {'Supabase' if supabase_enabled() else 'CSV'}")
    page = st.radio("画面", ["学習", "筆記", "穴埋め", "復習", "AI追加", "登録"], horizontal=True, label_visibility="collapsed")
    if page == "学習":
        study_screen(df)
    elif page == "筆記":
        quiz_screen(df, "written")
    elif page == "穴埋め":
        quiz_screen(df, "cloze")
    elif page == "復習":
        review_screen(df)
    elif page == "AI追加":
        ai_screen(df)
    else:
        register_screen(df)


if __name__ == "__main__":
    main()
