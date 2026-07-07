"""Helper utilities for Evolve: Adaptive Priority Engine.

Pure-Python helpers with **no internal dependencies** so they can be imported
by every other module without risking circular imports.
"""
from __future__ import annotations

import os
import re
import json
import hashlib
from datetime import datetime, timezone

# --------------------------------------------------------------------------- #
# Regex patterns
# --------------------------------------------------------------------------- #
# Matches ```lang\n ... ``` fenced code blocks (language optional).
CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*\n(.*?)```", re.DOTALL)
# Matches inline citation markers like [Frame #4] or [Cluster #2].
CITATION_RE = re.compile(r"\[(?:Frame|Cluster)\s*#?(\d+)\]", re.IGNORECASE)

# --------------------------------------------------------------------------- #
# Language -> file extension map for the workspace tooling
# --------------------------------------------------------------------------- #
LANG_EXT = {
    "python": "py", "py": "py", "python3": "py",
    "javascript": "js", "js": "js", "node": "js",
    "typescript": "ts", "ts": "ts",
    "jsx": "jsx", "tsx": "tsx",
    "html": "html", "htm": "html",
    "css": "css", "scss": "scss", "sass": "sass",
    "json": "json", "json5": "json",
    "yaml": "yaml", "yml": "yaml",
    "toml": "toml", "ini": "ini",
    "xml": "xml", "markdown": "md", "md": "md",
    "bash": "sh", "shell": "sh", "sh": "sh", "zsh": "sh",
    "powershell": "ps1", "bat": "bat",
    "sql": "sql",
    "go": "go", "golang": "go",
    "rust": "rs", "rs": "rs",
    "java": "java", "kotlin": "kt", "kt": "kt",
    "c": "c", "cpp": "cpp", "c++": "cpp", "h": "h",
    "ruby": "rb", "rb": "rb",
    "php": "php",
    "swift": "swift",
    "scala": "scala",
    "r": "r", "dart": "dart",
    "dockerfile": "dockerfile",
    "text": "txt", "txt": "txt", "plaintext": "txt", "": "txt",
}

WORKSPACE_DIR = os.path.join(os.getcwd(), "workspace")


# --------------------------------------------------------------------------- #
# Time helpers
# --------------------------------------------------------------------------- #
def now_iso() -> str:
    """UTC ISO-8601 timestamp (used for persistence)."""
    return datetime.now(timezone.utc).isoformat()


def now_local() -> str:
    """Local ISO-8601 timestamp."""
    return datetime.now().astimezone().isoformat()


def current_time() -> str:
    """Human-readable local timestamp."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def short_time(ts: str) -> str:
    """Render an ISO timestamp as HH:MM for compact UI display."""
    if not ts:
        return "--:--"
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt.strftime("%H:%M")
    except Exception:
        return ts[:5] if len(ts) >= 5 else "--:--"


def relative_day_phrase(ts_iso: str | None) -> str:
    """Friendly relative time (e.g. '4m ago', '2h ago', '3d ago')."""
    if not ts_iso:
        return "—"
    try:
        dt = datetime.fromisoformat(ts_iso.replace("Z", "+00:00"))
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        secs = (now - dt).total_seconds()
    except Exception:
        return "—"
    if secs < 0:
        return "just now"
    if secs < 60:
        return "just now"
    if secs < 3600:
        return f"{int(secs // 60)}m ago"
    if secs < 86400:
        return f"{int(secs // 3600)}h ago"
    return f"{int(secs // 86400)}d ago"


# --------------------------------------------------------------------------- #
# Code & citation parsing
# --------------------------------------------------------------------------- #
def extract_code_blocks(text: str) -> list[dict]:
    """Extract fenced code blocks from Markdown text.

    Returns a list of ``{"language", "code", "index"}`` dicts.
    """
    blocks: list[dict] = []
    if not text:
        return blocks
    for i, m in enumerate(CODE_BLOCK_RE.finditer(text)):
        lang = (m.group(1) or "text").strip().lower() or "text"
        code = m.group(2).rstrip()
        if code:
            blocks.append({"language": lang, "code": code, "index": i})
    return blocks


def parse_citations(text: str) -> list[int]:
    """Return the ordered, de-duplicated list of citation numbers in *text*."""
    if not text:
        return []
    seen: list[int] = []
    for m in CITATION_RE.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def lang_extension(lang: str) -> str:
    """Map a language label to a file extension (defaults to txt)."""
    key = (lang or "txt").strip().lower()
    if not key:
        key = "txt"
    # Special-case Dockerfile which has no conventional extension separator.
    if key in {"dockerfile"}:
        return "dockerfile"
    return LANG_EXT.get(key, "txt")


# --------------------------------------------------------------------------- #
# Workspace (file system) tooling
# --------------------------------------------------------------------------- #
def save_snippet(code: str, language: str, index: int = 0) -> str:
    """Persist a code snippet into the local ``/workspace`` directory.

    Creates the directory if needed and returns the absolute file path.
    """
    os.makedirs(WORKSPACE_DIR, exist_ok=True)
    ext = lang_extension(language)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_idx = re.sub(r"[^0-9a-zA-Z_-]", "_", str(index))
    fname = f"snippet_{stamp}_{safe_idx}.{ext}"
    path = os.path.join(WORKSPACE_DIR, fname)

    if ext == "py":
        header = (
            f"# Evolve Workspace Snippet\n"
            f"# Language: {language}\n"
            f"# Saved: {current_time()}\n\n"
        )
    else:
        header = (
            f"// Evolve Workspace Snippet\n"
            f"// Language: {language}\n"
            f"// Saved: {current_time()}\n\n"
        )

    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + code + "\n")
    except OSError:
        # Last-resort: try a .txt fallback name.
        path = os.path.join(WORKSPACE_DIR, f"snippet_{stamp}_{safe_idx}.txt")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(header + code + "\n")
    return path


def list_workspace_files() -> list[dict]:
    """List files currently in the workspace directory."""
    if not os.path.isdir(WORKSPACE_DIR):
        return []
    files: list[dict] = []
    for name in sorted(os.listdir(WORKSPACE_DIR)):
        full = os.path.join(WORKSPACE_DIR, name)
        if os.path.isfile(full):
            try:
                size = os.path.getsize(full)
            except OSError:
                size = 0
            files.append({"name": name, "path": full, "size": size})
    return files


# --------------------------------------------------------------------------- #
# Generic helpers
# --------------------------------------------------------------------------- #
def truncate(text: str, n: int = 240) -> str:
    """Trim *text* to at most *n* characters, appending an ellipsis."""
    if not text:
        return ""
    text = text.strip()
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def hash_id(text: str) -> str:
    """Stable 12-char hex id for arbitrary text."""
    return hashlib.sha1((text or "").encode("utf-8")).hexdigest()[:12]


def safe_json_loads(raw, default=None):
    """Lenient json.loads that never raises."""
    if default is None:
        default = {}
    if not raw:
        return default
    try:
        return json.loads(raw)
    except Exception:
        return default


def safe_json_dumps(obj) -> str:
    """Lenient json.dumps with a string fallback."""
    try:
        return json.dumps(obj, ensure_ascii=False, default=str)
    except Exception:
        return "{}"


def env_or(key: str, default):
    """Return env var *key* or *default* (treating empty string as missing)."""
    val = os.environ.get(key)
    return val if val not in (None, "") else default


def bool_env(key: str, default: bool = False) -> bool:
    """Parse a boolean env var."""
    val = os.environ.get(key)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}
