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


def test_no_fp32_is_a_flag_and_not_a_suite():
    from evals.run import split_flags

    assert split_flags(["perf", "--no-fp32"]) == (["perf"], {"--no-fp32"})
    assert split_flags([]) == ([], set())
    with pytest.raises(SystemExit):
        split_flags(["perf", "--fast"])


def test_git_sha_falls_back_when_git_is_missing(monkeypatch):
    import subprocess

    from evals.run import git_sha

    def no_git(*args, **kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", no_git)
    assert git_sha() == "nogit"


def test_results_folder_follows_bgr_results(monkeypatch, tmp_path):
    import importlib

    import evals.run

    monkeypatch.setenv("BGR_RESULTS", str(tmp_path))
    assert importlib.reload(evals.run).RESULTS == tmp_path
    monkeypatch.delenv("BGR_RESULTS")
    importlib.reload(evals.run)
