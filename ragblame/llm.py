"""Ask Claude through the local Claude Code CLI, with a disk cache, retries and a thread pool.

Every call runs `claude -p <prompt> --model <model> --tools ""`. The CLI uses whatever login it
already has. Optionally set CLAUDE_CODE_OAUTH_TOKEN, or set RAGBLAME_KEYCHAIN_SERVICE to the name of
a macOS Keychain item that holds a Claude Code token. The token is held in memory only and passed
to the CLI through its environment; it is never printed, logged or written.

Replies are cached at <home>/cache/llm/<sha256(model + "\\n" + prompt)>.json, where <home> is
$RAGBLAME_HOME or <repo>/.ragblame. Only successful replies are cached.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

TIMEOUT = 240                 # seconds per CLI call
BACKOFF = (2, 4, 8)           # seconds to wait before each retry
JSON_NOTE = "\n\nReply with only the JSON object."
TOKEN_VAR = "CLAUDE_CODE_OAUTH_TOKEN"
REPO_ROOT = Path(__file__).resolve().parent.parent


class LLMError(RuntimeError):
    """A Claude call that still failed after its retries."""


def _run_subprocess(cmd, env, timeout):
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    return r.returncode, r.stdout, r.stderr


# Tests replace these two.
_runner = _run_subprocess     # (cmd, env, timeout) -> (returncode, stdout, stderr)
_sleep = time.sleep

_lock = threading.Lock()
_tokens: dict[str, str] = {}  # keychain service -> token, in memory only
_calls = 0


def calls_made() -> int:
    """How many times this process has run the Claude CLI (retries count, cache hits don't)."""
    return _calls


def home() -> Path:
    h = os.environ.get("RAGBLAME_HOME")
    return Path(h) if h else REPO_ROOT / ".ragblame"


def _model(model: str | None) -> str:
    return model or os.environ.get("RAGBLAME_MODEL") or "sonnet"


def _read_keychain(service: str) -> str:
    r = subprocess.run(["security", "find-generic-password", "-s", service, "-w"],
                       capture_output=True, text=True)
    return r.stdout.strip()


def _environ() -> tuple[dict, str | None]:
    """The CLI's environment, and the token we added to it (so errors can redact it)."""
    env = dict(os.environ)
    service = env.get("RAGBLAME_KEYCHAIN_SERVICE")
    if service and not env.get(TOKEN_VAR):
        with _lock:
            if service not in _tokens:
                _tokens[service] = _read_keychain(service)
            tok = _tokens[service]
        if tok:
            env[TOKEN_VAR] = tok
            return env, tok
    return env, env.get(TOKEN_VAR) or None


def _cache_path(model: str, prompt: str) -> Path:
    key = hashlib.sha256((model + "\n" + prompt).encode()).hexdigest()
    return home() / "cache" / "llm" / f"{key}.json"


def _read_cache(path: Path) -> str | None:
    try:
        reply = json.loads(path.read_text())["reply"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return reply if isinstance(reply, str) and reply.strip() else None


def _write_cache(path: Path, model: str, prompt: str, reply: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"model": model, "prompt_sha": hashlib.sha256(prompt.encode()).hexdigest(), "reply": reply}
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    tmp.write_text(json.dumps(entry, ensure_ascii=False))
    os.replace(tmp, path)


def _call(prompt: str, model: str) -> str:
    """Run the CLI, retrying failures; returns the stripped reply or raises LLMError."""
    global _calls
    cmd = ["claude", "-p", prompt, "--model", model, "--tools", ""]
    env, token = _environ()
    problem = ""
    for attempt in range(len(BACKOFF) + 1):
        if attempt:
            _sleep(BACKOFF[attempt - 1])
        with _lock:
            _calls += 1
        try:
            code, out, err = _runner(cmd, env, TIMEOUT)
        except subprocess.TimeoutExpired:
            problem = f"timed out after {TIMEOUT}s"
            continue
        except FileNotFoundError:
            raise LLMError("the `claude` CLI was not found on PATH") from None
        reply = (out or "").strip()
        if code == 0 and reply:
            return reply
        problem = f"exit {code}: " + ((err or "").strip() or reply or "empty reply")[-400:]
    if token:
        problem = problem.replace(token, "[token]")
    raise LLMError(f"claude failed after {len(BACKOFF)} retries ({problem})")


def _ask(prompt: str, model: str, parse=None):
    """Cached call. `parse` may raise ValueError, in which case the reply counts as a failure
    and is not cached."""
    path = _cache_path(model, prompt)
    cached = _read_cache(path)
    if cached is not None:
        try:
            return parse(cached) if parse else cached
        except ValueError:
            pass                              # a stale bad entry: ask again
    reply = _call(prompt, model)
    value = parse(reply) if parse else reply
    _write_cache(path, model, prompt, reply)
    return value


def ask(prompt: str, *, model: str | None = None) -> str:
    """Claude's reply to `prompt`, from the cache when possible."""
    return _ask(prompt, _model(model))


def extract_json(text: str) -> dict:
    """The first top-level {...} object in `text`, ignoring prose and ```json fences."""
    decoder = json.JSONDecoder()
    i = text.find("{")
    while i != -1:
        try:
            obj, _ = decoder.raw_decode(text, i)
        except ValueError:
            obj = None
        if isinstance(obj, dict):
            return obj
        i = text.find("{", i + 1)
    raise ValueError("no JSON object in reply: " + text[:200])


def ask_json(prompt: str, *, model: str | None = None) -> dict:
    """Claude's reply parsed as a JSON object. One extra try asks for only the object."""
    m = _model(model)
    try:
        return _ask(prompt, m, extract_json)
    except ValueError:
        pass
    try:
        return _ask(prompt + JSON_NOTE, m, extract_json)
    except ValueError as e:
        raise LLMError(str(e)) from None


def map_concurrent(fn, items, workers: int = 4) -> list:
    """[fn(x) for x in items] on a thread pool, in input order. A call that raises leaves its
    exception object (usually an LLMError) in that slot instead of a result."""
    items = list(items)
    if not items:
        return []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(items)))) as pool:
        futures = [pool.submit(fn, x) for x in items]
        out = []
        for f in futures:
            try:
                out.append(f.result())
            except Exception as e:  # noqa: BLE001 - the caller decides what a failed slot means
                out.append(e)
        return out
