from types import SimpleNamespace

import httpx
import openai
import pytest

from board_game_reco.llm import JUDGE_MODEL, LLM, Retryable, _flagged, judge_client, judge_json, load_dotenv


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


def rate_limit(retry_after: str | None) -> openai.RateLimitError:
    headers = {"retry-after": retry_after} if retry_after else {}
    response = httpx.Response(429, headers=headers, request=httpx.Request("POST", "https://api.groq.com"))
    return openai.RateLimitError("rate limited", response=response, body=None)


def test_judge_json_uses_thinking_mode_sampling_and_a_strict_schema(monkeypatch):
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    completions = FakeCompletions('{"score": 4, "rationale": "fits"}')
    reply, tokens = judge_json(client(completions), "sys", "user", {"type": "object"}, "relevance")
    assert reply == {"score": 4, "rationale": "fits"}
    assert tokens == 18
    sent = completions.kwargs[0]
    assert sent["model"] == JUDGE_MODEL
    assert sent["temperature"] == 1.0 and sent["top_p"] == 0.95
    assert sent["reasoning_effort"] == "low"
    assert sent["extra_body"] == {"reasoning_format": "hidden"}
    assert sent["response_format"]["json_schema"]["strict"] is True
    assert "stream" not in sent


def test_judge_json_turns_a_rate_limit_into_retryable_with_the_server_wait():
    with pytest.raises(Retryable) as caught:
        judge_json(client(FakeCompletions(error=rate_limit("7"))), "s", "u", {}, "n")
    assert caught.value.retry_after == 7.0


def bad_request(code):
    import httpx
    import openai

    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"))
    return openai.BadRequestError("bad", response=response, body={"code": code})


def test_judge_json_retries_when_the_model_breaks_the_schema():
    with pytest.raises(Retryable):
        judge_json(client(FakeCompletions(error=bad_request("json_validate_failed"))), "s", "u", {}, "n")


def test_judge_json_raises_other_bad_requests():
    import openai

    with pytest.raises(openai.BadRequestError):
        judge_json(client(FakeCompletions(error=bad_request("model_not_found"))), "s", "u", {}, "n")


def test_judge_json_raises_on_a_reply_that_is_not_json():
    with pytest.raises(ValueError):
        judge_json(client(FakeCompletions("not json")), "s", "u", {}, "n")


def test_judge_client_is_none_without_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    assert judge_client(tmp_path / ".env") is None
