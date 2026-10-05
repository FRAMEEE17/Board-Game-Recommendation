# tests/test_run.py
import pytest

from evals.run import NETWORK, OFFLINE, select


def test_no_arguments_runs_only_the_suites_that_need_no_key():
    assert select([]) == ["parse", "route", "recommend", "perf"]
    assert not set(select([])) & set(NETWORK)


def test_network_suites_run_only_when_named():
    assert select(["judge", "relevance"]) == ["judge", "relevance"]
    assert set(OFFLINE) & set(NETWORK) == set()


def test_an_unknown_suite_is_refused():
    with pytest.raises(SystemExit):
        select(["bogus"])
