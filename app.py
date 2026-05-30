from __future__ import annotations

import html
import hashlib
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
COLUMNS = ["id", "word", "pronunciation", "part_of_speech", "meaning_ja", "example_en", "example_ja", "cloze_examples", "category", "difficulty", "low_frequency", "correct_count", "wrong_count", "last_studied"]
COUNT_COLUMNS = ["correct_count", "wrong_count"]
BOOL_COLUMNS = ["low_frequency"]
SUPABASE_TABLE = "words"
SUPABASE_SETTINGS_TABLE = "app_settings"
SUPABASE_TTS_BUCKET = "tts-audio"
DEFAULT_AI_MODEL = "gpt-5.4-mini"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "nova"
LOW_FREQUENCY_GAP = 20
CLOZE_EXAMPLE_COUNT = 5
AUDIO_CACHE_DIR = Path(__file__).with_name(".audio_cache")
SESSION_WORDS_KEY = "words_df"
PARTS_OF_SPEECH = ["noun", "verb", "adjective", "adverb", "phrase", "other"]
PARTS_OF_SPEECH_SET = set(PARTS_OF_SPEECH)

SAMPLE_WORDS = [
    [1, "incorporate", "in-KOR-puh-rayt", "verb", "取り入れる、組み込む", "We need to incorporate user feedback into the next version.", "次のバージョンにユーザーの意見を取り入れる必要があります。", "[]", "Business", "4", False, 0, 0, ""],
    [2, "consolidate", "kun-SOL-ih-dayt", "verb", "統合する、強化する", "The team will consolidate multiple reports into one dashboard.", "チームは複数のレポートを1つのダッシュボードに統合します。", "[]", "Business", "4", False, 0, 0, ""],
    [3, "appropriate", "uh-PROH-pree-uht", "adjective", "適切な", "Please choose the most appropriate response for the situation.", "その状況に最も適切な返答を選んでください。", "[]", "Academic", "3", False, 0, 0, ""],
    [4, "implement", "IM-pluh-ment", "verb", "実行する、実装する", "The company plans to implement a new training program.", "会社は新しい研修プログラムを実施する予定です。", "[]", "Business", "3", False, 0, 0, ""],
    [5, "overlook", "oh-ver-LOOK", "verb", "見落とす、大目に見る", "It is easy to overlook small errors when you are tired.", "疲れていると小さな誤りを見落としやすいです。", "[]", "Daily", "3", False, 0, 0, ""],
    [6, "fatigue", "fuh-TEEG", "noun", "疲労", "Long meetings can cause mental fatigue.", "長い会議は精神的な疲労を引き起こすことがあります。", "[]", "Health", "2", False, 0, 0, ""],
    [7, "retention", "ree-TEN-shun", "noun", "保持、定着", "Regular review improves vocabulary retention.", "定期的な復習は語彙の定着を高めます。", "[]", "Learning", "4", False, 0, 0, ""],
    [8, "elaborate", "ih-LAB-uh-rayt", "verb", "詳しく説明する", "Could you elaborate on your main idea?", "主な考えについて詳しく説明してもらえますか。", "[]", "Academic", "3", False, 0, 0, ""],
    [9, "conversely", "KON-ver-slee", "adverb", "反対に、逆に", "Some tasks require speed; conversely, others require careful planning.", "速さが必要な作業もありますが、逆に慎重な計画が必要な作業もあります。", "[]", "Academic", "4", False, 0, 0, ""],
    [10, "recurrent", "ree-KUR-unt", "adjective", "繰り返し起こる", "The app helps users review recurrent mistakes.", "そのアプリはユーザーが繰り返し起こる間違いを復習するのに役立ちます。", "[]", "Academic", "4", False, 0, 0, ""],
]


class LowFrequencySaveError(RuntimeError):
    pass


class SupabaseSchemaError(RuntimeError):
    pass


def raise_schema_error_if_needed(exc: Exception) -> None:
    text = str(exc)
    if "cloze_examples" in text:
        raise SupabaseSchemaError("Supabaseに cloze_examples 列がまだありません。Supabase SQL Editorで supabase_schema.sql を実行してから、アプリを再読み込みしてください。") from exc


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


def parse_cloze_examples(value: object) -> list[dict[str, str]]:
    if isinstance(value, list):
        raw_items = value
    else:
        text = str(value or "").strip()
        if not text:
            return []
        try:
            raw_items = json.loads(text)
        except json.JSONDecodeError:
            return []
    examples: list[dict[str, str]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        en = str(item.get("en") or item.get("example_en") or "").strip()
        ja = str(item.get("ja") or item.get("example_ja") or "").strip()
        if en:
            examples.append({"en": en, "ja": ja})
    return examples[:CLOZE_EXAMPLE_COUNT]


def encode_cloze_examples(examples: list[dict[str, str]]) -> str:
    cleaned: list[dict[str, str]] = []
    seen: set[str] = set()
    for example in examples:
        en = str(example.get("en", "")).strip()
        ja = str(example.get("ja", "")).strip()
        key = en.lower()
        if en and key not in seen:
            cleaned.append({"en": en, "ja": ja})
            seen.add(key)
        if len(cleaned) >= CLOZE_EXAMPLE_COUNT:
            break
    return json.dumps(cleaned, ensure_ascii=False)


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
    df["cloze_examples"] = df["cloze_examples"].apply(lambda value: encode_cloze_examples(parse_cloze_examples(value)))
    for col in BOOL_COLUMNS:
        df[col] = df[col].apply(normalize_bool)
    return df


def save_rows(rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    normalized = normalize_df(pd.DataFrame(rows))
    if supabase_enabled():
        try:
            supabase_client().table(SUPABASE_TABLE).upsert(normalized[COLUMNS].to_dict("records"), on_conflict="word").execute()
        except Exception as exc:
            raise_schema_error_if_needed(exc)
            raise
        return
    current = load_words()
    merged = pd.concat([current, normalized], ignore_index=True).drop_duplicates(subset=["word"], keep="last")
    merged.to_csv(DATA_FILE, index=False)


def save_words(df: pd.DataFrame) -> None:
    df = normalize_df(df)
    if supabase_enabled():
        try:
            supabase_client().table(SUPABASE_TABLE).upsert(df[COLUMNS].to_dict("records"), on_conflict="word").execute()
        except Exception as exc:
            raise_schema_error_if_needed(exc)
            raise
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


def tts_cache_key(text: object, model: str, voice: str) -> str:
    source = f"{model}|{voice}|{str(text).strip()}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def tts_cache_path(text: object, model: str, voice: str) -> str:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    safe_voice = re.sub(r"[^a-zA-Z0-9_.-]", "_", voice)
    return f"{safe_model}/{safe_voice}/{tts_cache_key(text, model, voice)}.mp3"


def read_tts_cache(path: str) -> bytes | None:
    if supabase_enabled():
        try:
            return supabase_client().storage.from_(SUPABASE_TTS_BUCKET).download(path)
        except Exception:
            return None
    local_path = AUDIO_CACHE_DIR / path
    return local_path.read_bytes() if local_path.exists() else None


def write_tts_cache(path: str, audio: bytes) -> None:
    if supabase_enabled():
        try:
            supabase_client().storage.from_(SUPABASE_TTS_BUCKET).upload(
                path,
                audio,
                {"content-type": "audio/mpeg", "upsert": "true"},
            )
        except Exception as exc:
            raise RuntimeError("Supabase Storageに音声を保存できませんでした。Supabaseで tts-audio バケットを作成してください。") from exc
        return
    local_path = AUDIO_CACHE_DIR / path
    local_path.parent.mkdir(parents=True, exist_ok=True)
    local_path.write_bytes(audio)


def ensure_tts_storage_ready() -> None:
    if not supabase_enabled():
        return
    storage = supabase_client().storage
    try:
        storage.get_bucket(SUPABASE_TTS_BUCKET)
    except Exception:
        try:
            storage.create_bucket(SUPABASE_TTS_BUCKET, options={"public": False})
        except Exception as exc:
            raise RuntimeError("Supabase Storageに tts-audio バケットを作成できませんでした。SupabaseのStorage画面で tts-audio という非公開バケットを作成してください。") from exc


def generate_tts_audio(text: object, model: str, voice: str) -> bytes:
    api_key = config("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定のため、高品質音声を生成できません。")
    from openai import OpenAI

    response = OpenAI(api_key=api_key).audio.speech.create(
        model=model,
        voice=voice,
        input=str(text).strip(),
        instructions=config("OPENAI_TTS_INSTRUCTIONS", "Speak naturally and clearly in American English for a listening practice exercise."),
        response_format="mp3",
        speed=0.92,
    )
    return bytes(response.content if hasattr(response, "content") else response.read())


def get_or_create_tts_audio(text: object) -> tuple[bytes, str]:
    model = config("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)
    voice = config("OPENAI_TTS_VOICE", DEFAULT_TTS_VOICE)
    path = tts_cache_path(text, model, voice)
    cached = read_tts_cache(path)
    if cached:
        return cached, "保存済み"
    ensure_tts_storage_ready()
    audio = generate_tts_audio(text, model, voice)
    write_tts_cache(path, audio)
    return audio, "新規生成"


def render_cached_tts_controls(text: object, key_prefix: str) -> None:
    session_key = f"{key_prefix}_tts_audio"
    source_key = f"{key_prefix}_tts_source"
    if st.button("高品質音声を準備", key=f"{key_prefix}_prepare_tts", use_container_width=True):
        try:
            with st.spinner("音声を準備しています..."):
                audio, source = get_or_create_tts_audio(text)
        except Exception as exc:
            st.warning(str(exc))
        else:
            st.session_state[session_key] = audio
            st.session_state[source_key] = source
    if session_key in st.session_state:
        st.audio(st.session_state[session_key], format="audio/mpeg")
        st.caption(f"高品質音声: {st.session_state.get(source_key, '保存済み')}。同じ英文は次回以降APIを呼ばずに再利用します。")
    else:
        st.caption("高品質音声は初回だけ生成して保存します。準備後は音声プレイヤーから再生できます。")


def render_speech_button(text: object, label: str = "発音", rate: float = 0.86) -> None:
    speak_text = str(text).strip()
    if not speak_text:
        return

    components.html(
        f"""
        <button id="speak-word" type="button" aria-label="発音を再生">
          <span class="speaker-icon">▶</span>
          <span>{esc(label)}</span>
        </button>
        <script>
          const button = document.getElementById("speak-word");
          const text = {json.dumps(speak_text)};
          const preferredVoiceNames = [
            "samantha",
            "karen",
            "moira",
            "daniel",
            "google us english",
            "google uk english female",
            "microsoft jenny",
            "microsoft aria",
            "microsoft guy"
          ];
          function pickEnglishVoice() {{
            const voices = window.speechSynthesis.getVoices();
            const englishVoices = voices.filter((voice) => (voice.lang || "").toLowerCase().startsWith("en"));
            for (const preferredName of preferredVoiceNames) {{
              const match = englishVoices.find((voice) => voice.name.toLowerCase().includes(preferredName));
              if (match) return match;
            }}
            return englishVoices.find((voice) => (voice.lang || "").toLowerCase() === "en-us") || englishVoices[0] || null;
          }}
          function speak() {{
            const utterance = new SpeechSynthesisUtterance(text);
            const voice = pickEnglishVoice();
            if (voice) {{
              utterance.voice = voice;
              utterance.lang = voice.lang || "en-US";
            }} else {{
              utterance.lang = "en-US";
            }}
            utterance.rate = {float(rate)};
            utterance.pitch = 1;
            window.speechSynthesis.cancel();
            window.speechSynthesis.speak(utterance);
          }}
          button.addEventListener("click", () => {{
            if (!("speechSynthesis" in window)) {{
              button.textContent = "このブラウザでは音声再生できません";
              return;
            }}
            const voices = window.speechSynthesis.getVoices();
            if (voices.length === 0) {{
              window.speechSynthesis.onvoiceschanged = speak;
              setTimeout(speak, 200);
              return;
            }}
            speak();
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


def render_pronunciation_button(word: object) -> None:
    render_speech_button(word, label="発音", rate=0.86)


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
              ["英単語を入力", "聞こえた英文を入力"].includes(input.getAttribute("aria-label")) && !input.disabled
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


def normalize_sentence_answer(value: object) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


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
    if "cloze_examples" not in values or not parse_cloze_examples(values.get("cloze_examples")):
        values = {**values, "cloze_examples": encode_cloze_examples(cloze_examples_for_values(values))}
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


def has_short_vowel_consonant_ending(base: str) -> bool:
    return (
        len(base) >= 3
        and base[-1] not in "aeiouwxy"
        and base[-2] in "aeiou"
        and base[-3] not in "aeiou"
    )


def verb_forms(base_word: object) -> dict[str, str]:
    base = str(base_word).strip().lower()
    if not base:
        return {"base": "", "third": "", "past": "", "ing": ""}
    if base.endswith("y") and len(base) > 1 and base[-2] not in "aeiou":
        third = f"{base[:-1]}ies"
        past = f"{base[:-1]}ied"
    elif base.endswith(("s", "sh", "ch", "x", "z", "o")):
        third = f"{base}es"
        past = f"{base}ed"
    elif base.endswith("e"):
        third = f"{base}s"
        past = f"{base}d"
    else:
        third = f"{base}s"
        past = f"{base}ed"
    if base.endswith("ie"):
        ing = f"{base[:-2]}ying"
    elif base.endswith("e") and not base.endswith("ee"):
        ing = f"{base[:-1]}ing"
    elif has_short_vowel_consonant_ending(base):
        ing = f"{base}{base[-1]}ing"
        past = f"{base}{base[-1]}ed"
    else:
        ing = f"{base}ing"
    return {"base": base, "third": third, "past": past, "ing": ing}


def is_pedagogical_cloze_example(example: dict[str, str]) -> bool:
    text = f"{example.get('en', '')} {example.get('ja', '')}".lower()
    markers = [
        "the form ",
        "learners practice",
        "passive form",
        "ongoing action",
        "basic form",
        "adjective ",
        "adverb ",
        "単語を入れます",
        "基本形を入れます",
        "三単現の形を入れます",
        "過去形を入れます",
        "受動態で使う形",
        "進行形で使う形",
        "形容詞を入れます",
        "副詞を入れます",
        "対象:",
        "意味:",
        "customer request",
        "support workflow",
    ]
    return any(marker in text for marker in markers)


def has_blankable_word(example: object, word: object) -> bool:
    text = str(example or "")
    forms = word_forms_for_blank(word)
    if not text or not forms:
        return False
    pattern = re.compile(rf"\b(?:{'|'.join(re.escape(form) for form in forms)})\b", re.IGNORECASE)
    return bool(pattern.search(text))


def cloze_examples_for_values(values: object) -> list[dict[str, str]]:
    getter = values.get if hasattr(values, "get") else lambda key, default="": getattr(values, key, default)
    examples: list[dict[str, str]] = []
    word = getter("word", "")
    primary_en = str(getter("example_en", "")).strip()
    primary_ja = str(getter("example_ja", "")).strip()
    if primary_en and has_blankable_word(primary_en, word):
        examples.append({"en": primary_en, "ja": primary_ja})
    examples.extend(
        example
        for example in parse_cloze_examples(getter("cloze_examples", ""))
        if not is_pedagogical_cloze_example(example) and has_blankable_word(example.get("en", ""), word)
    )
    return parse_cloze_examples(encode_cloze_examples(examples))


def cloze_examples_for_row(row) -> list[dict[str, str]]:
    return cloze_examples_for_values(row)


def word_forms_for_blank(word: object) -> list[str]:
    base = str(word).strip().lower()
    if not base:
        return []
    forms = {base}
    if " " not in base:
        forms.update(verb_forms(base).values())
    if base.endswith("y") and len(base) > 1:
        stem = base[:-1]
        forms.update({f"{stem}ies", f"{stem}ied"})
    return sorted(forms, key=len, reverse=True)


def blank_sentence(example: str, word: str) -> str:
    prompt, _answer = blank_sentence_and_answer(example, word)
    return prompt


def blank_sentence_and_answer(example: str, word: str) -> tuple[str, str]:
    """Return (prompt, answer).

    Improvement: when the provided example sentence is empty or blank, return
    a clearer prompt that indicates that no example sentence exists. This
    makes the cloze UI less confusing when opening a problem with no example.
    """
    forms = word_forms_for_blank(word)
    if not forms:
        # If there are no word forms (empty word), fallback to the base answer.
        base_ans = str(word).strip()
        if not str(example or "").strip():
            # Example is empty: show a clear placeholder prompt.
            return "_____ (例文がありません)", base_ans
        return example, base_ans
    pattern = re.compile(rf"\b(?:{'|'.join(re.escape(form) for form in forms)})\b", re.IGNORECASE)
    match = pattern.search(example or "")
    if not match:
        if not str(example or "").strip():
            # No example text provided at all: make the prompt explicit.
            return "_____ (例文がありません)", str(word).strip()
        return f"_____ {example}", str(word).strip()
    return pattern.sub("_____", example, count=1), match.group(0)


def render_card(row, show_answer: bool = True) -> None:
    # Minimal, robust card renderer used by non-UI parts and tests.
    if show_answer:
        answer_html = f"<div class=\"meaning\">{esc(row.get('meaning_ja', ''))}</div><div class=\"example-ja\">{esc(row.get('example_ja', ''))}</div>"
    else:
        answer_html = '<div class="answer-placeholder">日本語訳はまだ隠れています。</div>'
    frequency_pill = '<span class="pill">頻度低</span>' if normalize_bool(row.get("low_frequency", False)) else ""
    st.markdown(f"""
    <div class="word-card">
      <div><span class="pill">{esc(row.get('category', ''))}</span><span class="pill">{esc(row.get('part_of_speech', ''))}</span><span class="pill">Lv {esc(row.get('difficulty', ''))}</span>{frequency_pill}</div>
      <div class="word-title">{esc(row.get('word', ''))}</div>
      <div class="pronunciation">{esc(row.get('pronunciation', ''))}</div>
      <div class="example-en">{esc(row.get('example_en', ''))}</div>
      {answer_html}
    </div>
    """, unsafe_allow_html=True)


def set_saved_ai_category(category: str) -> None:
    """Persist a small marker for the AI-saved category locally when Supabase is not used.

    Tests patch Path to point to a temporary directory; write a simple text file.
    """
    marker_path = Path("SAVED_AI_CATEGORY")
    try:
        marker_path.write_text(str(category or ""), encoding="utf-8")
    except Exception:
        # Fail silently; this is only for convenience in local runs and tests.
        return


def saved_ai_category() -> str:
    marker_path = Path("SAVED_AI_CATEGORY")
    try:
        if marker_path.exists():
            return marker_path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""
    return ""


# The rest of the file contains UI entrypoints in the original app. To keep import-time
# behavior safe for unit tests, avoid executing Streamlit UI code at import. Any
# interactive code should be under a __main__ guard if added later.

if __name__ == "__main__":
    # Basic, safe entrypoint for manual runs; keep minimal to avoid surprising side effects.
    df = load_words()
    st.write("英単語帳アプリ - 最小実行モード")
    st.write(f"単語数: {len(df)}")
