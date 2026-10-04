import pytest

from board_game_reco.llm import Retryable
from board_game_reco.recommender import grounded
from evals.judge import DailyLimit, Judge, Pacer, game_facts


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.slept: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


class FakeCall:
    def __init__(self, replies) -> None:
        self.replies = list(replies)

    def __call__(self, system, user, schema, name):
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 500


def make_judge(tmp_path, replies):
    clock = Clock()
    judge = Judge(FakeCall(replies), tmp_path, Pacer(clock=clock, sleep=clock.sleep), sleep=clock.sleep)
    return judge, clock


def test_pacer_waits_when_the_token_window_is_full():
    clock = Clock()
    pacer = Pacer(tokens_per_minute=8000, requests_per_minute=30, clock=clock, sleep=clock.sleep)
    pacer.wait(5000)
    pacer.record(5000)
    clock.now = 10.0
    pacer.wait(5000)
    assert clock.slept == [50.0]


def test_pacer_waits_when_the_request_window_is_full():
    clock = Clock()
    pacer = Pacer(tokens_per_minute=8000, requests_per_minute=2, clock=clock, sleep=clock.sleep)
    for _ in range(2):
        pacer.wait(10)
        pacer.record(10)
    pacer.wait(10)
    assert clock.slept == [60.0]


def test_scores_are_cached_by_query_game_prompt_version_and_run(tmp_path, catalog):
    ok = {"score": 4, "rationale": "fits"}
    judge, _ = make_judge(tmp_path, [ok, ok, ok, ok])
    catan, pandemic = catalog.find("CATAN"), catalog.find("Pandemic")
    first = judge.relevance("a trading game", catan)
    again = judge.relevance("a trading game", catan)
    assert (first.cached, again.cached, again.score) == (False, True, 4)
    judge.relevance("a trading game", pandemic)
    judge.faithfulness("facts", "reason")
    judge.faithfulness("facts", "reason", run=1)
    assert judge.calls == 4 and judge.hits == 1


def test_a_rate_limit_is_retried_after_the_server_wait(tmp_path, catalog):
    judge, clock = make_judge(tmp_path, [Retryable(3.0), {"score": 2, "rationale": "loose"}])
    assert judge.relevance("a trading game", catalog.find("CATAN")).score == 2
    assert clock.slept == [3.0]


def test_a_long_wait_stops_with_daily_limit_and_caches_nothing(tmp_path, catalog):
    judge, _ = make_judge(tmp_path, [Retryable(3600.0)])
    with pytest.raises(DailyLimit):
        judge.relevance("a trading game", catalog.find("CATAN"))
    assert list(tmp_path.glob("*.json")) == []


def test_an_out_of_range_score_is_rejected_and_not_cached(tmp_path, catalog):
    judge, _ = make_judge(tmp_path, [{"score": 9, "rationale": "x"}])
    with pytest.raises(ValueError):
        judge.relevance("a trading game", catalog.find("CATAN"))
    assert list(tmp_path.glob("*.json")) == []


def test_game_facts_hold_every_number_the_template_quotes(catalog):
    g = catalog.find("CATAN")
    reason = (f"{g.name} ({g.year}) plays {g.min_players}-{g.max_players} players in about {g.max_minutes} minutes. "
              f"Ages {g.min_age}+, weight {g.complexity:.1f} out of 5, rated {g.rating:.1f} by {g.num_ratings:,} people.")
    assert grounded(reason, g, game_facts(g))
