import json
import subprocess
import threading

import pytest

from ragblame import llm

TOKEN = "sk-ant-oat-FAKE-test-token-123"


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("RAGBLAME_HOME", str(tmp_path / "home"))
    monkeypatch.delenv("RAGBLAME_MODEL", raising=False)
    monkeypatch.delenv("RAGBLAME_KEYCHAIN_SERVICE", raising=False)
    monkeypatch.setattr(llm, "_sleep", lambda s: None)
    monkeypatch.setattr(llm, "_tokens", {})
    return tmp_path / "home"


class FakeRunner:
    """Replays a list of (returncode, stdout, stderr) results, or exceptions, one per call."""

    def __init__(self, *results):
        self.results = list(results)
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self, cmd, env, timeout):
        with self.lock:
            self.calls.append((cmd, env, timeout))
            r = self.results.pop(0) if len(self.results) > 1 else self.results[0]
        if isinstance(r, BaseException):
            raise r
        return r


def test_ask_builds_the_claude_command(monkeypatch):
    run = FakeRunner((0, "hi\n", ""))
    monkeypatch.setattr(llm, "_runner", run)
    assert llm.ask("say hi") == "hi"
    cmd, _, timeout = run.calls[0]
    assert cmd == ["claude", "-p", "say hi", "--model", "sonnet", "--tools", ""]
    assert timeout == 240


def test_model_defaults_to_env(monkeypatch):
    monkeypatch.setenv("RAGBLAME_MODEL", "opus")
    run = FakeRunner((0, "hi", ""))
    monkeypatch.setattr(llm, "_runner", run)
    llm.ask("say hi")
    assert run.calls[0][0][4] == "opus"


def test_cache_returns_without_calling_runner(monkeypatch, isolated):
    run = FakeRunner((0, "first", ""))
    monkeypatch.setattr(llm, "_runner", run)
    assert llm.ask("q", model="sonnet") == "first"
    files = list((isolated / "cache" / "llm").glob("*.json"))
    assert len(files) == 1
    entry = json.loads(files[0].read_text())
    assert set(entry) == {"model", "prompt_sha", "reply"}

    def boom(cmd, env, timeout):
        raise AssertionError("runner must not be called on a cache hit")

    monkeypatch.setattr(llm, "_runner", boom)
    before = llm.calls_made()
    assert llm.ask("q", model="sonnet") == "first"
    assert llm.calls_made() == before


def test_cache_key_includes_model(monkeypatch):
    run = FakeRunner((0, "a", ""), (0, "b", ""))
    monkeypatch.setattr(llm, "_runner", run)
    assert llm.ask("q", model="sonnet") == "a"
    assert llm.ask("q", model="opus") == "b"


def test_retries_then_raises(monkeypatch):
    sleeps = []
    monkeypatch.setattr(llm, "_sleep", sleeps.append)
    run = FakeRunner((1, "", "rate limited"), subprocess.TimeoutExpired("claude", 240), (0, "  \n", ""),
                     (1, "", "rate limited"))
    monkeypatch.setattr(llm, "_runner", run)
    before = llm.calls_made()
    with pytest.raises(llm.LLMError) as e:
        llm.ask("q")
    assert len(run.calls) == 4          # one try plus 3 retries
    assert sleeps == [2, 4, 8]
    assert llm.calls_made() - before == 4
    assert "rate limited" in str(e.value)


def test_retry_succeeds_after_failure(monkeypatch):
    run = FakeRunner((1, "", "overloaded"), (0, "ok", ""))
    monkeypatch.setattr(llm, "_runner", run)
    assert llm.ask("q") == "ok"
    assert len(run.calls) == 2


def test_failure_not_cached(monkeypatch, isolated):
    monkeypatch.setattr(llm, "_runner", FakeRunner((1, "", "boom")))
    with pytest.raises(llm.LLMError):
        llm.ask("q")
    assert not list(isolated.rglob("*.json"))
    run = FakeRunner((0, "recovered", ""))
    monkeypatch.setattr(llm, "_runner", run)
    assert llm.ask("q") == "recovered"
    assert len(run.calls) == 1


def test_ask_json_extracts_object_from_prose(monkeypatch):
    reply = 'Sure! Here is the result: {"questions": [{"q": "a {b} c"}], "n": 1} Hope that helps {not json}.'
    monkeypatch.setattr(llm, "_runner", FakeRunner((0, reply, "")))
    assert llm.ask_json("q") == {"questions": [{"q": "a {b} c"}], "n": 1}


def test_ask_json_skips_braces_in_prose_before_the_object(monkeypatch):
    reply = 'Use {name} as a placeholder.\n{"ok": true}'
    monkeypatch.setattr(llm, "_runner", FakeRunner((0, reply, "")))
    assert llm.ask_json("q") == {"ok": True}


def test_ask_json_handles_fence(monkeypatch):
    reply = 'Here you go:\n```json\n{\n  "a": [1, 2],\n  "b": {"c": "d"}\n}\n```\n'
    monkeypatch.setattr(llm, "_runner", FakeRunner((0, reply, "")))
    assert llm.ask_json("q") == {"a": [1, 2], "b": {"c": "d"}}


def test_ask_json_retries_once_with_note_then_raises(monkeypatch, isolated):
    run = FakeRunner((0, "no json here", ""))
    monkeypatch.setattr(llm, "_runner", run)
    with pytest.raises(llm.LLMError):
        llm.ask_json("q")
    prompts = [c[0][2] for c in run.calls]
    assert prompts == ["q", "q\n\nReply with only the JSON object."]
    assert not list(isolated.rglob("*.json"))   # non-JSON replies are failures: never cached


def test_ask_json_note_retry_can_succeed(monkeypatch):
    run = FakeRunner((0, "prose only", ""), (0, '{"x": 1}', ""))
    monkeypatch.setattr(llm, "_runner", run)
    assert llm.ask_json("q") == {"x": 1}


def test_map_concurrent_keeps_order_and_captures_errors():
    def fn(x):
        if x == 3:
            raise llm.LLMError("three failed")
        if x == 5:
            raise ValueError("five")
        return x * 10

    out = llm.map_concurrent(fn, list(range(8)), workers=4)
    assert [o for i, o in enumerate(out) if i not in (3, 5)] == [0, 10, 20, 40, 60, 70]
    assert isinstance(out[3], llm.LLMError) and str(out[3]) == "three failed"
    assert isinstance(out[5], ValueError)


def test_token_never_in_cache_file(monkeypatch, isolated):
    monkeypatch.setenv("RAGBLAME_KEYCHAIN_SERVICE", "test.service")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    looked_up = []
    monkeypatch.setattr(llm, "_read_keychain", lambda s: looked_up.append(s) or TOKEN)
    run = FakeRunner((0, "hello", ""))
    monkeypatch.setattr(llm, "_runner", run)

    assert llm.ask("q") == "hello"
    cmd, env, _ = run.calls[0]
    assert env["CLAUDE_CODE_OAUTH_TOKEN"] == TOKEN
    assert TOKEN not in " ".join(cmd)
    assert looked_up == ["test.service"]
    for f in isolated.rglob("*"):
        if f.is_file():
            assert TOKEN not in f.read_text()

    llm.ask("q2")
    assert looked_up == ["test.service"]   # read once, then held in memory


def test_token_redacted_from_errors(monkeypatch):
    monkeypatch.setenv("RAGBLAME_KEYCHAIN_SERVICE", "test.service")
    monkeypatch.delenv("CLAUDE_CODE_OAUTH_TOKEN", raising=False)
    monkeypatch.setattr(llm, "_read_keychain", lambda s: TOKEN)
    monkeypatch.setattr(llm, "_runner", FakeRunner((1, "", f"bad token {TOKEN}")))
    with pytest.raises(llm.LLMError) as e:
        llm.ask("q")
    assert TOKEN not in str(e.value)


def test_existing_token_env_is_not_overridden(monkeypatch):
    monkeypatch.setenv("RAGBLAME_KEYCHAIN_SERVICE", "test.service")
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "from-env")

    def no_keychain(s):
        raise AssertionError("keychain must not be read when the env var is set")

    monkeypatch.setattr(llm, "_read_keychain", no_keychain)
    run = FakeRunner((0, "hi", ""))
    monkeypatch.setattr(llm, "_runner", run)
    llm.ask("q")
    assert run.calls[0][1]["CLAUDE_CODE_OAUTH_TOKEN"] == "from-env"
