from __future__ import annotations

import html
import hashlib
import json
import os
import re
from datetime import date
from datetime import datetime
from datetime import timedelta
from difflib import SequenceMatcher
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

DATA_FILE = Path(__file__).with_name("words.csv")
STUDY_EVENTS_FILE = Path(__file__).with_name("study_events.csv")
COLUMNS = ["id", "word", "pronunciation", "part_of_speech", "meaning_ja", "example_en", "example_ja", "cloze_examples", "category", "difficulty", "low_frequency", "correct_count", "wrong_count", "last_studied"]
COUNT_COLUMNS = ["correct_count", "wrong_count"]
BOOL_COLUMNS = ["low_frequency"]
SUPABASE_TABLE = "words"
SUPABASE_STUDY_EVENTS_TABLE = "study_events"
SUPABASE_SETTINGS_TABLE = "app_settings"
SUPABASE_TTS_BUCKET = "tts-audio"
DEFAULT_AI_MODEL = "gpt-5.4-mini"
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"
DEFAULT_TTS_VOICE = "nova"
DEFAULT_TTS_ACCENT = "british"
DEFAULT_TTS_SPEED = 1.0
DEFAULT_TTS_INSTRUCTIONS = (
    "Speak at a natural native-speaker pace in clear British English. "
    "Use a natural British accent suitable for English listening practice. "
    "Keep the pronunciation crisp, conversational, and easy to shadow."
)
DEFAULT_TIMEZONE = "Asia/Tokyo"
LOW_FREQUENCY_GAP = 20
CLOZE_EXAMPLE_COUNT = 5
AUDIO_CACHE_DIR = Path(__file__).with_name(".audio_cache")
DEFAULT_DAILY_GOAL = 50
SESSION_WORDS_KEY = "words_df"
STUDY_EVENT_COLUMNS = ["word_id", "word", "mode", "correct", "studied_on", "studied_at"]
MODE_LABELS = {"study": "学習カード", "written": "筆記", "fill": "穴埋め", "listening": "聞き取り"}
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


def normalize_study_events(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=STUDY_EVENT_COLUMNS)
    df = df.copy()
    for col in STUDY_EVENT_COLUMNS:
        if col not in df.columns:
            df[col] = False if col == "correct" else 0 if col == "word_id" else ""
    df = df[STUDY_EVENT_COLUMNS].fillna("")
    df["word_id"] = pd.to_numeric(df["word_id"], errors="coerce").fillna(0).astype(int)
    df["correct"] = df["correct"].apply(normalize_bool)
    for col in ["word", "mode", "studied_on", "studied_at"]:
        df[col] = df[col].astype(str)
    return df


def load_study_events(limit: int | None = None) -> pd.DataFrame:
    if supabase_enabled():
        try:
            query = supabase_client().table(SUPABASE_STUDY_EVENTS_TABLE).select("*").order("studied_at", desc=True)
            if limit:
                query = query.limit(int(limit))
            rows = query.execute().data or []
            return normalize_study_events(pd.DataFrame(rows))
        except Exception:
            return normalize_study_events(pd.DataFrame())
    if not STUDY_EVENTS_FILE.exists():
        return normalize_study_events(pd.DataFrame())
    events = normalize_study_events(pd.read_csv(STUDY_EVENTS_FILE, keep_default_na=False))
    events = events.sort_values("studied_at", ascending=False)
    return events.head(int(limit)) if limit else events


def study_event_from_row(row: pd.Series | dict[str, object], mode: str, correct: bool) -> dict[str, object]:
    row_dict = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    now = datetime.now(app_timezone()).isoformat(timespec="seconds")
    return {
        "word_id": int(row_dict.get("id", 0) or 0),
        "word": str(row_dict.get("word", "")),
        "mode": str(mode),
        "correct": bool(correct),
        "studied_on": today(),
        "studied_at": now,
    }


def record_study_event(row: pd.Series | dict[str, object], mode: str, correct: bool) -> bool:
    event = study_event_from_row(row, mode, correct)
    if supabase_enabled():
        try:
            supabase_client().table(SUPABASE_STUDY_EVENTS_TABLE).insert(event).execute()
            return True
        except Exception:
            return False
    events = load_study_events()
    updated = pd.concat([events, normalize_study_events(pd.DataFrame([event]))], ignore_index=True)
    updated[STUDY_EVENT_COLUMNS].to_csv(STUDY_EVENTS_FILE, index=False)
    return True


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


def tts_accent() -> str:
    return config("OPENAI_TTS_ACCENT", DEFAULT_TTS_ACCENT).lower()


def tts_speed() -> float:
    raw_speed = config("OPENAI_TTS_SPEED", str(DEFAULT_TTS_SPEED))
    try:
        speed = float(raw_speed)
    except ValueError:
        return DEFAULT_TTS_SPEED
    return speed if speed > 0 else DEFAULT_TTS_SPEED


def tts_instructions() -> str:
    return config("OPENAI_TTS_INSTRUCTIONS", DEFAULT_TTS_INSTRUCTIONS)


def tts_cache_key(text: object, model: str, voice: str, accent: str, speed: float, instructions: str) -> str:
    instructions_digest = hashlib.sha256(instructions.strip().encode("utf-8")).hexdigest()
    source = f"{model}|{voice}|{accent}|{speed:.2f}|{instructions_digest}|{str(text).strip()}".encode("utf-8")
    return hashlib.sha256(source).hexdigest()


def tts_cache_path(text: object, model: str, voice: str, accent: str, speed: float, instructions: str) -> str:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    safe_voice = re.sub(r"[^a-zA-Z0-9_.-]", "_", voice)
    safe_accent = re.sub(r"[^a-zA-Z0-9_.-]", "_", accent)
    return f"{safe_model}/{safe_voice}/{safe_accent}/{tts_cache_key(text, model, voice, accent, speed, instructions)}.mp3"


def read_tts_cache(path: str) -> bytes | None:
    if supabase_enabled():
        try:
            return supabase_client().storage.from_(SUPABASE_TTS_BUCKET).download(path)
        except Exception:
            return None
    local_path = AUDIO_CACHE_DIR / path
    return local_path.read_bytes() if local_path.exists() else None


def expected_tts_cache_paths(df: pd.DataFrame, model: str, voice: str, accent: str, speed: float, instructions: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for row in normalize_df(df).itertuples():
        for example in cloze_examples_for_row(row._asdict()):
            text = str(example.get("en", "")).strip()
            if not text:
                continue
            path = tts_cache_path(text, model, voice, accent, speed, instructions)
            if path in seen:
                continue
            seen.add(path)
            rows.append(
                {
                    "word": row.word,
                    "text": text,
                    "path": path,
                }
            )
    return pd.DataFrame(rows, columns=["word", "text", "path"])


def list_local_tts_cache_paths() -> dict[str, int]:
    if not AUDIO_CACHE_DIR.exists():
        return {}
    paths: dict[str, int] = {}
    for path in AUDIO_CACHE_DIR.rglob("*.mp3"):
        if path.is_file():
            paths[path.relative_to(AUDIO_CACHE_DIR).as_posix()] = path.stat().st_size
    return paths


def list_supabase_tts_cache_paths(model: str, voice: str, accent: str) -> dict[str, int]:
    safe_model = re.sub(r"[^a-zA-Z0-9_.-]", "_", model)
    safe_voice = re.sub(r"[^a-zA-Z0-9_.-]", "_", voice)
    safe_accent = re.sub(r"[^a-zA-Z0-9_.-]", "_", accent)
    prefix = f"{safe_model}/{safe_voice}/{safe_accent}"
    try:
        items = supabase_client().storage.from_(SUPABASE_TTS_BUCKET).list(prefix, {"limit": 1000})
    except Exception:
        return {}
    paths: dict[str, int] = {}
    for item in items or []:
        name = str(item.get("name", ""))
        if not name.endswith(".mp3"):
            continue
        metadata = item.get("metadata") or {}
        size = int(metadata.get("size") or 0)
        paths[f"{prefix}/{name}"] = size
    return paths


def tts_cache_inventory(df: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    model = config("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)
    voice = config("OPENAI_TTS_VOICE", DEFAULT_TTS_VOICE)
    accent = tts_accent()
    speed = tts_speed()
    instructions = tts_instructions()
    expected = expected_tts_cache_paths(df, model, voice, accent, speed, instructions)
    cached_paths = list_supabase_tts_cache_paths(model, voice, accent) if supabase_enabled() else list_local_tts_cache_paths()
    if expected.empty:
        return expected.assign(cached=[], size=[]), 0, len(cached_paths)
    inventory = expected.copy()
    inventory["cached"] = inventory["path"].isin(cached_paths)
    inventory["size"] = inventory["path"].map(cached_paths).fillna(0).astype(int)
    return inventory, int(inventory["cached"].sum()), len(cached_paths)


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


def generate_tts_audio(text: object, model: str, voice: str, instructions: str, speed: float) -> bytes:
    api_key = config("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が未設定のため、高品質音声を生成できません。")
    from openai import OpenAI

    response = OpenAI(api_key=api_key).audio.speech.create(
        model=model,
        voice=voice,
        input=str(text).strip(),
        instructions=instructions,
        response_format="mp3",
        speed=speed,
    )
    return bytes(response.content if hasattr(response, "content") else response.read())


def get_or_create_tts_audio(text: object) -> tuple[bytes, str]:
    model = config("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)
    voice = config("OPENAI_TTS_VOICE", DEFAULT_TTS_VOICE)
    accent = tts_accent()
    speed = tts_speed()
    instructions = tts_instructions()
    path = tts_cache_path(text, model, voice, accent, speed, instructions)
    cached = read_tts_cache(path)
    if cached:
        return cached, "保存済み"
    ensure_tts_storage_ready()
    audio = generate_tts_audio(text, model, voice, instructions, speed)
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


def app_timezone() -> ZoneInfo:
    try:
        return ZoneInfo(config("APP_TIMEZONE", DEFAULT_TIMEZONE) or DEFAULT_TIMEZONE)
    except Exception:
        return ZoneInfo(DEFAULT_TIMEZONE)


def today() -> str:
    return datetime.now(app_timezone()).date().isoformat()


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


def studied_date_strings(df: pd.DataFrame) -> set[str]:
    dates: set[str] = set()
    for value in df.get("last_studied", pd.Series(dtype=str)).astype(str):
        text = value.strip()
        if not text:
            continue
        try:
            dates.add(date.fromisoformat(text[:10]).isoformat())
        except ValueError:
            continue
    return dates


def normalized_studied_dates(studied_dates: set[str]) -> list[date]:
    dates: list[date] = []
    for value in studied_dates:
        try:
            dates.append(date.fromisoformat(str(value)[:10]))
        except ValueError:
            continue
    return sorted(set(dates))


def consecutive_learning_days(studied_dates: set[str], today_value: str | None = None) -> int:
    normalized_dates = set(normalized_studied_dates(studied_dates))
    if not normalized_dates:
        return 0
    current = date.fromisoformat(today_value or today())
    if current not in normalized_dates:
        yesterday = current - timedelta(days=1)
        if yesterday not in normalized_dates:
            return 0
        current = yesterday
    streak = 0
    while current in normalized_dates:
        streak += 1
        current -= timedelta(days=1)
    return streak


def longest_learning_streak(studied_dates: set[str]) -> int:
    dates = normalized_studied_dates(studied_dates)
    if not dates:
        return 0
    longest = 1
    current = 1
    for previous, current_date in zip(dates, dates[1:]):
        if current_date == previous + timedelta(days=1):
            current += 1
        else:
            longest = max(longest, current)
            current = 1
    return max(longest, current)


def latest_learning_date(studied_dates: set[str]) -> str:
    dates = normalized_studied_dates(studied_dates)
    return dates[-1].isoformat() if dates else "-"


def dashboard_stats(df: pd.DataFrame, events: pd.DataFrame | None = None, today_value: str | None = None, daily_goal: int = DEFAULT_DAILY_GOAL) -> dict[str, object]:
    work = with_scores(normalize_df(df))
    event_log = normalize_study_events(events) if events is not None else pd.DataFrame(columns=STUDY_EVENT_COLUMNS)
    has_event_log = not event_log.empty
    today_text = today_value or today()
    total_words = len(work)
    total_correct = int(work["_correct"].sum()) if not work.empty else 0
    total_wrong = int(work["_wrong"].sum()) if not work.empty else 0
    total_answers = total_correct + total_wrong
    attempted = (work["_correct"] + work["_wrong"]) > 0 if not work.empty else pd.Series(dtype=bool)
    if has_event_log:
        today_events = event_log[event_log["studied_on"].astype(str).str[:10] == today_text]
        today_count = int(len(today_events))
        today_correct = int(today_events["correct"].sum())
        today_wrong = today_count - today_correct
        studied_dates = set(event_log["studied_on"].astype(str).str[:10].loc[event_log["studied_on"].astype(str).str.strip() != ""])
    else:
        today_count = int((work["last_studied"].astype(str).str[:10] == today_text).sum()) if not work.empty else 0
        today_correct = 0
        today_wrong = 0
        studied_dates = studied_date_strings(work)
    weak_count = int((work["weakness_score"] > 0).sum()) if not work.empty else 0
    new_count = int((~attempted).sum()) if not work.empty else 0
    mastered_count = int(((work["_correct"] >= 3) & (work["weakness_score"] <= 0)).sum()) if not work.empty else 0
    goal = max(int(daily_goal), 1)
    return {
        "total_words": total_words,
        "today_count": today_count,
        "daily_goal": goal,
        "goal_percent": min(today_count / goal, 1.0),
        "accuracy": total_correct / total_answers if total_answers else 0.0,
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "today_correct": today_correct,
        "today_wrong": today_wrong,
        "weak_count": weak_count,
        "new_count": new_count,
        "mastered_count": mastered_count,
        "streak": consecutive_learning_days(studied_dates, today_text),
        "longest_streak": longest_learning_streak(studied_dates),
        "last_learning_date": latest_learning_date(studied_dates),
        "event_log_available": has_event_log,
    }


def daily_event_counts(events: pd.DataFrame, today_value: str | None = None, days: int = 14) -> pd.DataFrame:
    today_date = date.fromisoformat(today_value or today())
    start_date = today_date - timedelta(days=max(int(days), 1) - 1)
    labels = [(start_date + timedelta(days=offset)).isoformat() for offset in range(max(int(days), 1))]
    counts = dict.fromkeys(labels, 0)
    event_log = normalize_study_events(events)
    if not event_log.empty:
        grouped = event_log["studied_on"].astype(str).str[:10].value_counts().to_dict()
        for label in labels:
            counts[label] = int(grouped.get(label, 0))
    return pd.DataFrame({"日付": labels, "学習回数": [counts[label] for label in labels]})


def recent_wrong_word_ranking(events: pd.DataFrame, today_value: str | None = None, days: int = 3, limit: int = 5) -> pd.DataFrame:
    event_log = normalize_study_events(events)
    if event_log.empty:
        return pd.DataFrame(columns=["word", "wrong_count", "last_wrong_on"])
    today_date = date.fromisoformat(today_value or today())
    start_date = today_date - timedelta(days=max(int(days), 1) - 1)
    wrong_events = event_log[
        (~event_log["correct"])
        & (pd.to_datetime(event_log["studied_on"], errors="coerce").dt.date >= start_date)
        & (pd.to_datetime(event_log["studied_on"], errors="coerce").dt.date <= today_date)
    ].copy()
    if wrong_events.empty:
        return pd.DataFrame(columns=["word", "wrong_count", "last_wrong_on"])
    ranking = (
        wrong_events.groupby("word", as_index=False)
        .agg(wrong_count=("correct", "size"), last_wrong_on=("studied_on", "max"))
        .sort_values(["wrong_count", "last_wrong_on", "word"], ascending=[False, False, True])
        .head(max(int(limit), 1))
    )
    return ranking[["word", "wrong_count", "last_wrong_on"]]


def mode_accuracy_summary(events: pd.DataFrame) -> pd.DataFrame:
    event_log = normalize_study_events(events)
    if event_log.empty:
        return pd.DataFrame(columns=["mode", "mode_label", "answers", "correct", "wrong", "accuracy_percent"])
    summary = (
        event_log.groupby("mode", as_index=False)
        .agg(answers=("correct", "size"), correct=("correct", "sum"))
        .sort_values(["answers", "mode"], ascending=[False, True])
    )
    summary["correct"] = summary["correct"].astype(int)
    summary["wrong"] = summary["answers"].astype(int) - summary["correct"]
    summary["accuracy_percent"] = (summary["correct"] / summary["answers"] * 100).round().astype(int)
    summary["mode_label"] = summary["mode"].map(MODE_LABELS).fillna(summary["mode"])
    return summary[["mode", "mode_label", "answers", "correct", "wrong", "accuracy_percent"]]


def daily_goal_achieved(stats: dict[str, object]) -> bool:
    return int(stats.get("today_count", 0)) >= int(stats.get("daily_goal", DEFAULT_DAILY_GOAL))


def daily_goal_message(stats: dict[str, object]) -> str:
    return f"今日の目標達成: {int(stats['today_count'])} / {int(stats['daily_goal'])}回"


def daily_goal_remaining(stats: dict[str, object]) -> int:
    return max(int(stats.get("daily_goal", DEFAULT_DAILY_GOAL)) - int(stats.get("today_count", 0)), 0)


def daily_goal_remaining_message(stats: dict[str, object]) -> str:
    return f"今日の目標まであと {daily_goal_remaining(stats)} 回"


def render_daily_goal_status(df: pd.DataFrame) -> None:
    stats = dashboard_stats(df, load_study_events())
    if daily_goal_achieved(stats):
        title = daily_goal_message(stats)
        note = "今日の学習目標を達成しました。続ける場合も、この記録は残ります。"
        class_name = "goal-achieved-banner"
    else:
        title = daily_goal_remaining_message(stats)
        note = f"現在 {int(stats['today_count'])} / {int(stats['daily_goal'])}回。少しずつ積み上げましょう。"
        class_name = "goal-remaining-banner"
    st.markdown(
        f"""
        <div class="{class_name}">
          <div class="goal-status-title">{esc(title)}</div>
          <div class="goal-status-note">{esc(note)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def priority(df: pd.DataFrame) -> pd.DataFrame:
    work = with_scores(df)
    work["_last"] = pd.to_datetime(work["last_studied"], errors="coerce").fillna(pd.Timestamp("1970-01-01"))
    return work.sort_values(["weakness_score", "_last", "_wrong", "word"], ascending=[False, True, False, True]).drop(columns=["_correct", "_wrong", "_last"])


def weak_words(df: pd.DataFrame) -> pd.DataFrame:
    ordered = priority(df)
    return ordered[ordered["weakness_score"] > 0]


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


def update_stats(df: pd.DataFrame, word_id: int, correct: bool, mode: str = "study") -> pd.DataFrame:
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
    record_study_event(row, mode, correct)
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
    forms = word_forms_for_blank(word)
    if not forms:
        return example, str(word).strip()
    pattern = re.compile(rf"\b(?:{'|'.join(re.escape(form) for form in forms)})\b", re.IGNORECASE)
    match = pattern.search(example)
    if not match:
        return f"_____ {example}", str(word).strip()
    return pattern.sub("_____", example, count=1), match.group(0)


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
            values = {
                "word": word.strip(),
                "pronunciation": pronunciation.strip(),
                "part_of_speech": part_of_speech,
                "meaning_ja": meaning.strip(),
                "example_en": example_en.strip(),
                "example_ja": example_ja.strip(),
                "category": category.strip() or "Uncategorized",
                "difficulty": difficulty,
                "low_frequency": low_frequency,
            }
            api_key = config("OPENAI_API_KEY")
            if api_key:
                try:
                    generated = generate_cloze_examples_with_ai([values], config("OPENAI_MODEL", DEFAULT_AI_MODEL), api_key)
                    examples = generated.get(word.strip().lower(), [])
                    if examples:
                        values["cloze_examples"] = encode_cloze_examples(examples)
                except Exception as exc:
                    st.warning(f"穴埋め例文のAI作成は失敗しました。単語は保存します: {exc}")
            try:
                df, created = upsert_word(df, values)
            except SupabaseSchemaError as exc:
                st.error(str(exc))
            else:
                st.success("新しい単語を登録しました。" if created else "既存の単語を更新しました。")
    with st.expander("登録済み単語"):
        st.dataframe(df[["word", "part_of_speech", "meaning_ja", "category", "difficulty", "low_frequency", "correct_count", "wrong_count", "last_studied"]].rename(columns={"word": "英単語", "part_of_speech": "品詞", "meaning_ja": "意味", "category": "カテゴリ", "difficulty": "難易度", "low_frequency": "頻度低", "correct_count": "正解", "wrong_count": "不正解", "last_studied": "最終学習日"}), use_container_width=True, hide_index=True)
    return df


def dashboard_metric(label: str, value: str, note: str = "") -> str:
    note_html = f'<div class="dashboard-note">{esc(note)}</div>' if note else ""
    return f"""
      <div class="dashboard-card">
        <div class="dashboard-label">{esc(label)}</div>
        <div class="dashboard-value">{esc(value)}</div>
        {note_html}
      </div>
    """


def dashboard_screen(df: pd.DataFrame) -> pd.DataFrame:
    events = load_study_events()
    stats = dashboard_stats(df, events)
    accuracy_percent = round(float(stats["accuracy"]) * 100)
    goal_percent = round(float(stats["goal_percent"]) * 100)
    today_note = f"正解 {stats['today_correct']} / 不正解 {stats['today_wrong']}" if stats["event_log_available"] else f"目標 {stats['daily_goal']}回"
    st.subheader("ダッシュボード")
    st.caption(f"学習の進み具合を、日本時間（{today()}）で集計します。学習ログがある場合、今日の学習は記録された解答数です。")
    st.markdown(
        f"""
        <div class="dashboard-grid">
          {dashboard_metric("総単語数", f"{stats['total_words']}語", "登録済み")}
          {dashboard_metric("今日の学習", f"{stats['today_count']}回", today_note)}
          {dashboard_metric("正解率", f"{accuracy_percent}%", f"正解 {stats['total_correct']} / 不正解 {stats['total_wrong']}")}
          {dashboard_metric("連続学習", f"{stats['streak']}日", f"最長 {stats['longest_streak']}日 / 最終 {stats['last_learning_date']}")}
          {dashboard_metric("苦手", f"{stats['weak_count']}語", f"全{stats['total_words']}語を評価")}
          {dashboard_metric("定着", f"{stats['mastered_count']}語", "正解3回以上")}
        </div>
        <div class="goal-panel">
          <div class="goal-row">
            <span>今日の目標</span>
            <strong>{stats['today_count']} / {stats['daily_goal']}回</strong>
          </div>
          <div class="progress-track"><div class="progress-fill" style="width:{goal_percent}%"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if not events.empty:
        daily_counts = daily_event_counts(events)
        chart_data = daily_counts.set_index("日付")
        st.markdown("#### 日別の学習回数")
        st.bar_chart(chart_data, height=220)
        total_recent = int(daily_counts["学習回数"].sum())
        active_days = int((daily_counts["学習回数"] > 0).sum())
        st.caption(f"直近14日: 合計 {total_recent}回 / 学習した日 {active_days}日")
        mode_summary = mode_accuracy_summary(events)
        if not mode_summary.empty:
            st.markdown("#### 問題形式ごとの正解率")
            st.dataframe(
                mode_summary[["mode_label", "accuracy_percent", "answers", "correct", "wrong"]].rename(
                    columns={
                        "mode_label": "形式",
                        "accuracy_percent": "正解率%",
                        "answers": "回答数",
                        "correct": "正解",
                        "wrong": "不正解",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
        wrong_ranking = recent_wrong_word_ranking(events)
        st.markdown("#### 最近よく間違える単語")
        if wrong_ranking.empty:
            st.info("直近3日に不正解だった単語はありません。")
        else:
            st.dataframe(
                wrong_ranking.rename(
                    columns={
                        "word": "英単語",
                        "wrong_count": "不正解回数",
                        "last_wrong_on": "最後に間違えた日",
                    }
                ),
                use_container_width=True,
                hide_index=True,
            )
    weak_word_rows = weak_words(df).head(5)
    if not weak_word_rows.empty:
        st.markdown("#### 次に減らしたい苦手")
        st.dataframe(
            weak_word_rows[["word", "meaning_ja", "weakness_score", "correct_count", "wrong_count", "last_studied"]].rename(
                columns={
                    "word": "英単語",
                    "meaning_ja": "意味",
                    "weakness_score": "苦手数",
                    "correct_count": "正解",
                    "wrong_count": "不正解",
                    "last_studied": "最終学習日",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
    if not events.empty:
        recent_events = events.head(10).copy()
        recent_events["mode_label"] = recent_events["mode"].map(MODE_LABELS).fillna(recent_events["mode"])
        recent_events["result_label"] = recent_events["correct"].map({True: "正解", False: "不正解"})
        st.markdown("#### 最近の学習ログ")
        st.dataframe(
            recent_events[["studied_at", "word", "mode_label", "result_label"]].rename(
                columns={
                    "studied_at": "日時",
                    "word": "英単語",
                    "mode_label": "形式",
                    "result_label": "結果",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
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
        df = update_stats(df, int(row["id"]), True, "study")
        st.session_state[history_key] = pushed_history(st.session_state.get(history_key, []), int(row["id"]))
        st.session_state[key] = next_id_for_session(df, int(row["id"]), recent_key)
        st.session_state[reveal_key] = False
        st.rerun()
    if c2.button("苦手", use_container_width=True):
        df = update_stats(df, int(row["id"]), False, "study")
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
    available = df if mode == "written" else df[df.apply(lambda row: bool(cloze_examples_for_row(row)), axis=1)]
    if available.empty:
        st.info("問題に使える単語がありません。AI追加画面で穴埋め例文を作り直すか、単語登録で英語の例文を入れてください。")
        return df
    current_key = f"{mode}_current_id"
    result_key = f"{mode}_result"
    answer_key = f"{mode}_answer"
    recent_key = f"{mode}_recent_ids"
    history_key = f"{mode}_history_ids"
    variant_key = f"{mode}_variant"
    variant_counts_key = f"{mode}_variant_counts"
    if current_key not in st.session_state or row_by_id(available, st.session_state[current_key]) is None:
        st.session_state[current_key] = next_id_for_session(available, None, recent_key)
    row = row_by_id(available, st.session_state[current_key])
    if mode == "written":
        prompt = row["meaning_ja"]
        expected_answer = str(row["word"])
        variant_hint = ""
    else:
        variants = cloze_examples_for_row(row)
        if st.session_state.get(variant_key, {}).get("id") != int(row["id"]):
            variant_counts = {
                int(key): int(value)
                for key, value in st.session_state.get(variant_counts_key, {}).items()
            }
            variant_index = variant_counts.get(int(row["id"]), 0) % max(len(variants), 1)
            variant_counts[int(row["id"])] = variant_index + 1
            st.session_state[variant_counts_key] = variant_counts
            st.session_state[variant_key] = {"id": int(row["id"]), "index": variant_index}
        variant_index = int(st.session_state.get(variant_key, {}).get("index", 0)) % max(len(variants), 1)
        variant = variants[variant_index]
        if mode == "listening":
            prompt = "音声を聞いて、聞こえた英文をすべて入力してください。"
            expected_answer = str(variant["en"])
        else:
            prompt, expected_answer = blank_sentence_and_answer(variant["en"], row["word"])
        variant_hint = variant.get("ja", "")
    hint = f"{row['part_of_speech']} ・ {row['category']} ・ Lv {row['difficulty']}" if mode == "written" else row["example_ja"]
    if mode == "listening":
        hint = "日本語訳は正解後に表示されます。"
    elif mode != "written" and variant_hint:
        hint = variant_hint
    titles = {"written": "筆記問題", "fill": "穴埋め問題", "listening": "聞き取り問題"}
    st.subheader(titles.get(mode, "問題"))
    st.caption("新しい単語を先に出し、1回目で正解が多い単語は後半へ、頻度低の単語は直近20回に出ている間は通常単語を優先します。")
    st.markdown(f'<div class="quiz-card"><div class="quiz-label">問題</div><div class="quiz-prompt">{esc(prompt)}</div><div class="hint-line">{esc(hint)}</div></div>', unsafe_allow_html=True)
    if mode == "listening":
        render_cached_tts_controls(expected_answer, f"{mode}_{int(row['id'])}_{variant_index}")
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
        if mode != "listening":
            render_pronunciation_button(result["expected"])
        if result["correct"]:
            if mode == "listening" and variant_hint:
                st.markdown(
                    f"""
                    <div class="answer-review">
                      <div class="answer-review-label">日本語訳</div>
                      <div class="answer-review-text">{esc(variant_hint)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            enable_return_to_next()
            st.caption("次へボタン、またはReturnキーで次の問題へ進めます。")
            history = st.session_state.get(history_key, [])
            if st.button("次へ", type="primary", use_container_width=True):
                st.session_state.pop(result_key, None)
                st.session_state.pop(variant_key, None)
                st.session_state[answer_key] = ""
                st.session_state[history_key] = pushed_history(history, int(row["id"]))
                st.session_state[current_key] = next_id_for_session(available, int(row["id"]), recent_key)
                st.rerun()
            if st.button("前へ", use_container_width=True, disabled=not bool(history)):
                previous_id, remaining_history = pop_previous_id(history, available)
                st.session_state.pop(result_key, None)
                st.session_state.pop(variant_key, None)
                st.session_state[answer_key] = ""
                st.session_state[history_key] = remaining_history
                if previous_id is not None:
                    st.session_state[current_key] = previous_id
                st.rerun()
            return df
        st.caption("もう一度入力して、正解できたら次へ進めます。")
    input_label = "聞こえた英文を入力" if mode == "listening" else "英単語を入力"
    with st.form(f"{mode}_form", clear_on_submit=True):
        answer = st.text_input(input_label, key=answer_key)
        submitted = st.form_submit_button("判定")
    focus_answer_input()
    if submitted:
        correct = normalize_sentence_answer(answer) == normalize_sentence_answer(expected_answer) if mode == "listening" else norm(answer) == norm(expected_answer)
        if is_first_quiz_attempt(st.session_state.get(result_key), int(row["id"])):
            df = update_stats(df, int(row["id"]), correct, mode)
        st.session_state[result_key] = {"id": int(row["id"]), "correct": correct, "expected": expected_answer, "answer": answer}
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


def render_tts_cache_status(df: pd.DataFrame) -> None:
    storage_label = f"Supabase Storage: {SUPABASE_TTS_BUCKET}" if supabase_enabled() else f"ローカル: {AUDIO_CACHE_DIR}"
    model = config("OPENAI_TTS_MODEL", DEFAULT_TTS_MODEL)
    voice = config("OPENAI_TTS_VOICE", DEFAULT_TTS_VOICE)
    accent = tts_accent()
    speed = tts_speed()
    with st.expander("高品質音声キャッシュ"):
        st.write("聞き取り問題で使う高品質音声が保存済みか確認できます。保存済みの英文は、次回以降OpenAI APIを呼ばずに再利用します。")
        st.caption(f"保存先: {storage_label} / モデル: {model} / Voice: {voice} / Accent: {accent} / Speed: {speed:.2f}")
        inventory, cached_count, stored_file_count = tts_cache_inventory(df)
        expected_count = len(inventory)
        c1, c2, c3 = st.columns(3)
        c1.metric("保存済み", f"{cached_count}件")
        c2.metric("対象英文", f"{expected_count}件")
        c3.metric("保存ファイル", f"{stored_file_count}件")
        if inventory.empty:
            st.info("穴埋め例文がまだないため、音声キャッシュの対象英文がありません。")
            return
        only_cached = st.checkbox("保存済みだけ表示", value=True)
        shown = inventory[inventory["cached"]] if only_cached else inventory
        if shown.empty:
            st.info("現在の穴埋め例文に対応する保存済み音声はまだありません。聞き取り問題で高品質音声を準備すると保存されます。")
            return
        table = shown.copy()
        table["status"] = table["cached"].map({True: "保存済み", False: "未生成"})
        table["size_kb"] = (table["size"] / 1024).round(1)
        st.dataframe(
            table[["word", "text", "status", "size_kb"]].rename(
                columns={
                    "word": "単語",
                    "text": "英文",
                    "status": "状態",
                    "size_kb": "サイズKB",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )


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


def sanitize_cloze_examples_for_word(word: object, examples: object) -> list[dict[str, str]]:
    return [
        example
        for example in parse_cloze_examples(examples)
        if not is_pedagogical_cloze_example(example) and has_blankable_word(example.get("en", ""), word)
    ][:CLOZE_EXAMPLE_COUNT]


def generate_cloze_examples_with_ai(rows: list[dict[str, object]], model: str, api_key: str) -> dict[str, list[dict[str, str]]]:
    if not rows:
        return {}
    from openai import OpenAI
    from pydantic import BaseModel, Field

    class ClozeExample(BaseModel):
        en: str = Field(description="Natural business or IT English sentence containing the target word or a natural inflected form")
        ja: str = Field(description="Natural Japanese translation of the full English sentence")

    class ClozeWord(BaseModel):
        word: str
        cloze_examples: list[ClozeExample] = Field(description="Exactly 5 natural cloze-practice sentences")

    class ClozeBatch(BaseModel):
        words: list[ClozeWord]

    lines = "\n".join(
        "- word: {word}; part_of_speech: {part}; meaning_ja: {meaning}; category: {category}; difficulty: {difficulty}; current example: {example_en}; current translation: {example_ja}".format(
            word=str(row.get("word", "")).strip(),
            part=str(row.get("part_of_speech", "")).strip(),
            meaning=str(row.get("meaning_ja", "")).strip(),
            category=str(row.get("category", "")).strip(),
            difficulty=str(row.get("difficulty", "")).strip(),
            example_en=str(row.get("example_en", "")).strip(),
            example_ja=str(row.get("example_ja", "")).strip(),
        )
        for row in rows
    )
    prompt = f"""
Create exactly 5 cloze-practice examples for each vocabulary entry below.

Rules:
- Use realistic business, product, engineering, support, analytics, meeting, or project contexts.
- The English sentence must be meaningful by itself and must contain the target word or a natural inflected form.
- For verbs, vary forms naturally: base, third-person singular, past, passive, progressive, or perfect when appropriate.
- The Japanese translation must translate the whole sentence naturally.
- Do not write grammar explanation sentences such as "The form ... is used".
- Do not include labels such as "meaning", "target", or "対象".
- Do not force the word into an unnatural customer request/support workflow template.
- Return one result for every input word.

Vocabulary:
{lines}
""".strip()
    parsed = OpenAI(api_key=api_key).responses.parse(
        model=model or DEFAULT_AI_MODEL,
        instructions="You are an English vocabulary coach for Japanese business and IT learners. Prioritize natural usage over grammar drills.",
        input=prompt,
        text_format=ClozeBatch,
    ).output_parsed

    examples_by_word: dict[str, list[dict[str, str]]] = {}
    requested_words = {str(row.get("word", "")).strip().lower() for row in rows}
    for item in parsed.words:
        word = item.word.strip()
        key = word.lower()
        if key not in requested_words:
            continue
        raw_examples = [
            {"en": example.en, "ja": example.ja}
            for example in item.cloze_examples
        ]
        examples = sanitize_cloze_examples_for_word(word, raw_examples)
        if examples:
            examples_by_word[key] = examples
    return examples_by_word


def update_cloze_examples(df: pd.DataFrame, examples_by_word: dict[str, list[dict[str, str]]]) -> tuple[pd.DataFrame, int]:
    if not examples_by_word:
        return df, 0
    updated = normalize_df(df)
    changed: list[int] = []
    for idx, row in updated.iterrows():
        word_key = str(row["word"]).strip().lower()
        examples = examples_by_word.get(word_key)
        if not examples:
            continue
        encoded = encode_cloze_examples(examples)
        if encoded != str(row.get("cloze_examples", "")):
            updated.at[idx, "cloze_examples"] = encoded
            changed.append(idx)
    if not changed:
        return set_words(updated), 0
    updated = normalize_df(updated)
    if supabase_enabled():
        save_rows(updated.loc[changed, COLUMNS].to_dict("records"))
    else:
        save_words(updated)
    return set_words(updated), len(changed)


def generate_ai_words(df: pd.DataFrame, count: int, category: str, difficulty: str, model: str) -> tuple[pd.DataFrame, list[str]]:
    api_key = config("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY が設定されていません。")
    from openai import OpenAI
    from pydantic import BaseModel, Field

    class ClozeExample(BaseModel):
        en: str = Field(description="English sentence containing the target word or an inflected form")
        ja: str = Field(description="Japanese translation of the sentence")

    class AiWord(BaseModel):
        word: str
        pronunciation: str
        part_of_speech: str = Field(description="Part of speech: noun, verb, adjective, adverb, phrase, or other")
        meaning_ja: str
        example_en: str
        example_ja: str
        cloze_examples: list[ClozeExample] = Field(description="Exactly 5 cloze-practice sentences")
        category: str
        difficulty: str = Field(pattern="^[1-5]$")

    class Batch(BaseModel):
        words: list[AiWord]

    existing = ", ".join(df["word"].astype(str).tolist()[:500])
    prompt = f"Generate {count} useful English vocabulary entries for a Japanese learner. Do not include these words: {existing}. Include part_of_speech as noun, verb, adjective, adverb, phrase, or other. Category hint: {category or 'any practical category'}. Difficulty hint: {difficulty}. Return Japanese meanings and translations. For each word, include exactly 5 cloze_examples. Each cloze example must contain the target word or a natural inflected form. For verbs, vary tense, active voice, passive voice, third-person singular, and progressive/perfect forms when natural."
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
        values["cloze_examples"] = encode_cloze_examples([
            {"en": example.get("en", ""), "ja": example.get("ja", "")}
            for example in values.get("cloze_examples", [])
            if isinstance(example, dict)
        ])
        values["cloze_examples"] = encode_cloze_examples(sanitize_cloze_examples_for_word(word, values["cloze_examples"]))
        if len(parse_cloze_examples(values["cloze_examples"])) < CLOZE_EXAMPLE_COUNT:
            values["cloze_examples"] = encode_cloze_examples(cloze_examples_for_values(values))
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
    with st.expander("穴埋め例文を作り直す"):
        st.write("既存単語すべてに、ビジネスやITの現場で自然に使える穴埋め例文を5件ずつ作成し直します。")
        if st.button("AIで穴埋め例文を全て作り直す", type="primary", use_container_width=True, disabled=not bool(api_key) or df.empty):
            try:
                all_examples: dict[str, list[dict[str, str]]] = {}
                records = df[COLUMNS].to_dict("records")
                with st.spinner("AIが穴埋め例文を作り直しています..."):
                    for start in range(0, len(records), 15):
                        batch = records[start:start + 15]
                        all_examples.update(generate_cloze_examples_with_ai(batch, model or model_default, api_key))
                    df, changed = update_cloze_examples(df, all_examples)
            except Exception as exc:
                st.error(f"穴埋め例文の作り直しに失敗しました: {exc}")
            else:
                st.success(f"{changed}語の穴埋め例文を更新しました。")
                st.rerun()
    render_tts_cache_status(df)
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
      div[data-testid="stSidebar"] { background:#fff; border-right:1px solid #e1e7f0; }
      div[data-testid="stSidebar"] div[data-testid="stRadio"] label { border-radius:8px; min-height:42px; padding:.3rem .45rem; }
      div[data-testid="stSidebar"] div[data-testid="stRadio"] label * { color:#172033!important; font-weight:800!important; }
      .word-card,.quiz-card { background:#fff; border:1px solid #e1e7f0; border-radius:8px; padding:1rem; box-shadow:0 8px 24px rgba(24,39,75,.08); margin:.4rem 0 1rem; }
      .pill { background:#eef6f1; color:#24533b; border:1px solid #d7ebdf; border-radius:999px; display:inline-flex; align-items:center; min-height:28px; padding:0 .65rem; font-size:.8rem; font-weight:700; margin-right:.4rem; }
      .word-title { color:#111827; font-size:2.2rem; font-weight:800; line-height:1.05; overflow-wrap:anywhere; margin-top:.8rem; }
      .pronunciation,.example-ja,.stats-line,.hint-line { color:#596579; font-size:.95rem; line-height:1.55; margin-top:.5rem; overflow-wrap:anywhere; }
      .meaning { color:#182033; font-size:1.15rem; font-weight:700; margin-top:1rem; overflow-wrap:anywhere; }
      .example-en { background:#f7f9fc; border-left:4px solid #2f6fed; color:#1f2937; margin-top:1rem; padding:.8rem; line-height:1.55; overflow-wrap:anywhere; }
      .answer-placeholder { background:#f7f9fc; border:1px dashed #cbd5e1; border-radius:8px; color:#687385; font-size:.95rem; margin-top:.85rem; padding:.75rem; text-align:center; }
      .dashboard-grid { display:grid; gap:.65rem; grid-template-columns:repeat(2,minmax(0,1fr)); margin:.75rem 0 1rem; }
      .dashboard-card { background:#fff; border:1px solid #e1e7f0; border-radius:8px; padding:.9rem; box-shadow:0 8px 24px rgba(24,39,75,.06); min-height:96px; }
      .dashboard-label { color:#596579; font-size:.82rem; font-weight:800; }
      .dashboard-value { color:#111827; font-size:1.65rem; font-weight:900; line-height:1.1; margin-top:.35rem; }
      .dashboard-note { color:#687385; font-size:.78rem; line-height:1.35; margin-top:.35rem; }
      .goal-panel { background:#fff; border:1px solid #e1e7f0; border-radius:8px; padding:1rem; margin:.35rem 0 1rem; }
      .goal-row { align-items:center; color:#172033; display:flex; justify-content:space-between; gap:.75rem; font-size:.95rem; }
      .goal-achieved-banner { background:#ecfdf5; border:1px solid #bbf7d0; border-radius:8px; box-shadow:0 8px 24px rgba(22,101,52,.08); margin:.75rem 0 1rem; padding:.9rem 1rem; }
      .goal-remaining-banner { background:#eff6ff; border:1px solid #bfdbfe; border-radius:8px; box-shadow:0 8px 24px rgba(37,99,235,.07); margin:.75rem 0 1rem; padding:.9rem 1rem; }
      .goal-status-title { color:#1e3a8a; font-size:1.05rem; font-weight:900; line-height:1.35; }
      .goal-status-note { color:#475569; font-size:.9rem; line-height:1.45; margin-top:.25rem; }
      .goal-achieved-banner .goal-status-title { color:#166534; }
      .goal-achieved-banner .goal-status-note { color:#24533b; }
      .progress-track { background:#e8edf5; border-radius:999px; height:12px; overflow:hidden; margin-top:.75rem; }
      .progress-fill { background:#2f6fed; border-radius:999px; height:100%; }
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
      @media (max-width: 430px) {
        .dashboard-grid { grid-template-columns:1fr; }
      }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    css()
    if not require_password():
        return
    st.title("英単語帳")
    df = words_for_session()

    menu = st.sidebar.radio("モード", ["ダッシュボード", "学習カード", "筆記問題", "穴埋め問題", "聞き取り問題", "復習", "単語登録", "AI追加"], index=0)
    st.sidebar.write(f"単語数: {len(df)}")
    render_daily_goal_status(df)

    if menu == "ダッシュボード":
        df = dashboard_screen(df)
    elif menu == "学習カード":
        df = study_screen(df)
    elif menu == "筆記問題":
        df = quiz_screen(df, "written")
    elif menu == "穴埋め問題":
        df = quiz_screen(df, "fill")
    elif menu == "聞き取り問題":
        df = quiz_screen(df, "listening")
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
