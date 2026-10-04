# tests/test_tune.py
import os
from pathlib import Path

from evals.tune import RECOMMENDER, latest_choice, write_defaults


def test_write_defaults_changes_only_the_similarity_and_rating_lines(tmp_path):
    copy = tmp_path / "recommender.py"
    original = RECOMMENDER.read_text(encoding="utf-8")
    copy.write_text(original, encoding="utf-8")
    write_defaults(copy, 1.4, 0.8)
    changed = copy.read_text(encoding="utf-8")
    assert "    similarity: float = 1.4\n" in changed
    assert "    rating: float = 0.8\n" in changed
    differing = [a for a, b in zip(original.splitlines(), changed.splitlines()) if a != b]
    assert len(differing) == 2


def test_latest_choice_reads_the_newest_run_that_tuned(tmp_path: Path):
    older, newer, parse_only = tmp_path / "2026-10-05_aaa", tmp_path / "2026-10-06_bbb", tmp_path / "2026-10-07_ccc"
    for folder, (similarity, rating) in ((older, (0.6, 0.2)), (newer, (1.4, 0.8))):
        folder.mkdir()
        (folder / "summary.csv").write_text(
            f"suite,metric,value\ntune,chosen_similarity,{similarity}\ntune,chosen_rating,{rating}\n")
    parse_only.mkdir()
    (parse_only / "summary.csv").write_text("suite,metric,value\nparse,field_accuracy_en,0.99\n")
    os.utime(older / "summary.csv", (1, 1))
    os.utime(newer / "summary.csv", (2, 2))
    assert latest_choice(tmp_path) == (1.4, 0.8)
