from types import SimpleNamespace

from board_game_reco.llm import LLM, _flagged, load_dotenv


class FakeCompletions:
    def __init__(self, content=None, error=None):
        self.content, self.error, self.kwargs = content, error, []

    def create(self, **kwargs):
        self.kwargs.append(kwargs)
        if self.error:
            raise self.error
        usage = SimpleNamespace(prompt_tokens=11, completion_tokens=7)
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)], usage=usage)


def client(completions):
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_json_sends_strict_schema_at_temperature_zero_and_counts_usage():
    completions = FakeCompletions('{"players": 4}')
    llm = LLM(client(completions))
    assert llm.json("sys", "user", {"type": "object"}, "req") == {"players": 4}
    sent = completions.kwargs[0]
    assert sent["temperature"] == 0
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert "stream" not in sent
    assert llm.usage == {"calls": 1, "prompt_tokens": 11, "completion_tokens": 7}


def test_any_failure_returns_none():
    assert LLM(client(FakeCompletions(error=TimeoutError()))).json("s", "u", {}, "n") is None
    assert LLM(client(FakeCompletions("not json"))).json("s", "u", {}, "n") is None
    assert LLM(client(FakeCompletions(error=RuntimeError()))).text("s", "u") is None
    assert LLM(client(FakeCompletions(error=RuntimeError()))).guard("x") is None


def test_flagged_reads_scores_and_labels():
    assert _flagged("0.98") is True
    assert _flagged("0.01") is False
    assert _flagged("MALICIOUS") is True
    assert _flagged("BENIGN") is False
    assert _flagged("???") is None


def test_from_env_returns_none_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert LLM.from_env(tmp_path / ".env") is None


def test_dotenv_never_overrides_real_environment(monkeypatch, tmp_path):
    env = tmp_path / ".env"
    env.write_text("LLM_MODEL=from-file\nLLM_BASE_URL='https://example.test'\n# comment\n")
    monkeypatch.setenv("LLM_MODEL", "from-env")
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    load_dotenv(env)
    import os
    assert os.environ["LLM_MODEL"] == "from-env"
    assert os.environ["LLM_BASE_URL"] == "https://example.test"
