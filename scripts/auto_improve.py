from __future__ import annotations

import argparse
import os
import re
from datetime import date
from pathlib import Path

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKLOG = ROOT / "IMPROVEMENT_BACKLOG.md"
HISTORY_FILE = ROOT / "AUTO_IMPROVEMENT_HISTORY.md"
DEFAULT_MODEL = "gpt-5-mini"
FALLBACK_MODELS = ("gpt-5-mini", "gpt-4.1-mini")

ALLOWED_EXACT_PATHS = {
    "app.py",
    "README.md",
    "IMPROVEMENT_BACKLOG.md",
    "AUTO_IMPROVEMENT_HISTORY.md",
}
ALLOWED_PREFIXES = ("docs/", "tests/")
BLOCKED_EXACT_PATHS = {
    ".streamlit/secrets.toml",
    "words.csv",
    "supabase_schema.sql",
}
BLOCKED_PREFIXES = (
    ".git/",
    ".github/workflows/",
    ".streamlit/",
)
BLOCKED_TASK_KEYWORDS = (
    "supabase schema",
    "schema破壊",
    "rls",
    "認証",
    "auth",
    "課金",
    "billing",
    "secret",
    "service role",
    "削除",
    "drop table",
    "truncate",
)
BLOCKED_CONTENT_PATTERNS = (
    r"\bDROP\s+TABLE\b",
    r"\bTRUNCATE\b",
    r"\bDELETE\s+FROM\b",
    r"\bALTER\s+POLICY\b",
    r"\bCREATE\s+POLICY\b",
    r"\bENABLE\s+ROW\s+LEVEL\s+SECURITY\b",
    r"sk-[A-Za-z0-9_-]{20,}",
    r"eyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}",
)


class FileChange(BaseModel):
    path: str = Field(description="Repository-relative path.")
    content: str = Field(description="Complete replacement content for this file.")


class ImprovementResult(BaseModel):
    summary: str
    safety: str
    tests: list[str]
    human_check: list[str]
    files: list[FileChange]


def repository_path(path_text: str) -> Path:
    path = Path(path_text)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path: {path_text}")
    return ROOT / path


def is_allowed_path(path_text: str) -> bool:
    normalized = path_text.replace("\\", "/")
    if normalized in BLOCKED_EXACT_PATHS:
        return False
    if any(normalized.startswith(prefix) for prefix in BLOCKED_PREFIXES):
        return False
    return normalized in ALLOWED_EXACT_PATHS or any(normalized.startswith(prefix) for prefix in ALLOWED_PREFIXES)


def ensure_safe_content(path_text: str, content: str) -> None:
    for pattern in BLOCKED_CONTENT_PATTERNS:
        if re.search(pattern, content, flags=re.IGNORECASE):
            raise ValueError(f"Blocked risky content in {path_text}: {pattern}")


def read_text_if_exists(path_text: str, max_chars: int = 32000) -> str:
    path = ROOT / path_text
    if not path.exists() or not path.is_file():
        return ""
    text = path.read_text(encoding="utf-8")
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[truncated]\n"
    return text


def safe_task(line: str) -> bool:
    lowered = line.lower()
    return not any(keyword in lowered for keyword in BLOCKED_TASK_KEYWORDS)


def recent_history_tasks(history: Path, limit: int = 20) -> set[str]:
    if not history.exists():
        return set()
    tasks: list[str] = []
    for line in history.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*-\s*\d{4}-\d{2}-\d{2}:\s*(.+)", line)
        if match:
            tasks.append(match.group(1).split(" — ", 1)[0].strip())
    return set(tasks[-limit:])


def selection_offset() -> int:
    raw = os.getenv("AUTO_IMPROVE_TASK_OFFSET") or os.getenv("GITHUB_RUN_ID") or ""
    if raw.isdigit():
        return int(raw)
    return date.today().toordinal() - 1


def pick_task(backlog: Path, history: Path = HISTORY_FILE, offset: int | None = None) -> str:
    tasks: list[str] = []
    for line in backlog.read_text(encoding="utf-8").splitlines():
        match = re.match(r"\s*-\s*\[\s\]\s*(.+)", line)
        if match and safe_task(match.group(1)):
            tasks.append(match.group(1).strip())
    if not tasks:
        raise RuntimeError("No safe unchecked task found in IMPROVEMENT_BACKLOG.md")
    recent = recent_history_tasks(history)
    fresh_tasks = [task for task in tasks if task not in recent]
    candidates = fresh_tasks or tasks
    return candidates[(selection_offset() if offset is None else offset) % len(candidates)]


def mark_task_done(backlog: Path, task: str) -> None:
    lines = backlog.read_text(encoding="utf-8").splitlines()
    target = f"- [ ] {task}"
    for index, line in enumerate(lines):
        if line.strip() == target:
            indent = line[: len(line) - len(line.lstrip())]
            lines[index] = f"{indent}- [x] {task}"
            backlog.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return
    raise RuntimeError(f"Selected task was not found in backlog: {task}")


def append_history(history: Path, task: str, result: ImprovementResult) -> None:
    if history.exists():
        text = history.read_text(encoding="utf-8").rstrip()
    else:
        text = "# 自動改善履歴\n\nマージ済みの自動改善PRで扱ったタスクを記録します。"
    entry = f"- {date.today().isoformat()}: {task} — {result.summary}"
    history.write_text(f"{text}\n{entry}\n", encoding="utf-8")


def build_context(task: str) -> str:
    files = [
        "README.md",
        "app.py",
        "tests/test_smoke.py",
        "IMPROVEMENT_BACKLOG.md",
        "AUTO_IMPROVEMENT_HISTORY.md",
    ]
    sections = [f"Selected task:\n{task}\n"]
    for path in files:
        content = read_text_if_exists(path)
        if content:
            sections.append(f"\n--- {path} ---\n{content}")
    return "\n".join(sections)


def apply_result(result: ImprovementResult) -> None:
    if not result.files:
        raise RuntimeError("AI returned no file changes.")
    if len(result.files) > 2:
        raise RuntimeError("AI returned too many file changes. Keep one PR small.")
    for change in result.files:
        if not is_allowed_path(change.path):
            raise ValueError(f"Path is outside the safe allowlist: {change.path}")
        ensure_safe_content(change.path, change.content)
    for change in result.files:
        target = repository_path(change.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(change.content, encoding="utf-8")


def pr_body(task: str, result: ImprovementResult | None, no_change_reason: str = "") -> str:
    if result is None:
        return (
            "## 改善内容\n"
            f"- 対象タスク: {task}\n"
            f"- 変更なし: {no_change_reason}\n\n"
            "## 安全性\n"
            "- 本番ブランチへ直接pushしていません。\n"
            "- secretsやSupabaseの破壊的変更は行っていません。\n\n"
            "## テスト\n"
            "- 変更なしのため未実行\n\n"
            "## 人間が確認する点\n"
            "- GitHub Actionsのログを確認してください。\n"
        )
    tests = "\n".join(f"- {item}" for item in result.tests) or "- GitHub Actionsでsmoke testを実行"
    checks = "\n".join(f"- {item}" for item in result.human_check) or "- スマホ画面で表示を確認"
    return (
        "## 改善内容\n"
        f"- 対象タスク: {task}\n"
        f"- {result.summary}\n\n"
        "## なぜ安全か\n"
        f"- {result.safety}\n"
        "- 自動変更できるファイルはallowlistで制限しています。\n"
        "- Supabase schema、RLS、認証、secrets、本番データは自動変更対象外です。\n\n"
        "## 実行したテスト結果\n"
        f"{tests}\n\n"
        "## 人間が確認すべき点\n"
        f"{checks}\n"
    )


def write_optional(path_text: str | None, content: str) -> None:
    if not path_text:
        return
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def candidate_models(model: str) -> list[str]:
    models = [model] if model else []
    models.extend(FALLBACK_MODELS)
    return list(dict.fromkeys(models))


def run_ai_once(task: str, model: str, api_key: str) -> ImprovementResult:
    from openai import OpenAI

    prompt = build_context(task)
    client = OpenAI(api_key=api_key)
    response = client.responses.parse(
        model=model,
        instructions=(
            "You are maintaining a small Streamlit vocabulary app. "
            "Make one safe, reviewable improvement for the selected task. "
            "Return complete replacement file contents only for changed files. "
            "Do not change Supabase schema, RLS, auth, secrets, production data, billing, "
            "or GitHub workflow files. Keep the PR tiny. Prefer tests or small UI/logic improvements."
        ),
        input=prompt,
        text_format=ImprovementResult,
    )
    return response.output_parsed


def run_ai(task: str, model: str, api_key: str) -> ImprovementResult:
    errors: list[str] = []
    for candidate in candidate_models(model):
        try:
            print(f"Trying AI model: {candidate}")
            return run_ai_once(task, candidate, api_key)
        except Exception as exc:
            errors.append(f"{candidate}: {exc}")
            print(f"AI model failed: {candidate}: {exc}")
    raise RuntimeError("All AI model attempts failed. " + " | ".join(errors))


def main() -> int:
    parser = argparse.ArgumentParser(description="Create one safe AI-assisted improvement.")
    parser.add_argument("--backlog", default=str(DEFAULT_BACKLOG))
    parser.add_argument("--model", default=os.getenv("AUTO_IMPROVE_MODEL") or os.getenv("OPENAI_MODEL") or DEFAULT_MODEL)
    parser.add_argument("--summary-path", default=os.getenv("AUTO_IMPROVEMENT_SUMMARY_PATH", ""))
    parser.add_argument("--task-path", default=os.getenv("AUTO_IMPROVEMENT_TASK_PATH", ""))
    args = parser.parse_args()

    backlog = Path(args.backlog)
    task = pick_task(backlog)
    write_optional(args.task_path, task)

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        message = "OPENAI_API_KEY is not set, so no AI change was generated."
        print(message)
        write_optional(args.summary_path, pr_body(task, None, message))
        return 0

    try:
        result = run_ai(task, args.model, api_key)
        apply_result(result)
        mark_task_done(backlog, task)
        append_history(HISTORY_FILE, task, result)
    except Exception as exc:
        message = f"AI improvement was skipped safely: {exc}"
        print(message)
        write_optional(args.summary_path, pr_body(task, None, message))
        return 0

    write_optional(args.summary_path, pr_body(task, result))
    print(f"Applied AI improvement for task: {task}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
