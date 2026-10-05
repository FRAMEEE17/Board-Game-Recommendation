# tests/test_build.py
import json

import pytest

from board_game_reco import build
from board_game_reco.gamemap import load_map
from tests.conftest import FakeEncoder


class ArmEncoder(FakeEncoder):
    name = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/onnx/model_qint8_arm64.onnx"


@pytest.mark.parametrize("arch", ["aarch64", "arm64", "ARM64"])
def test_the_arm_file_matches_arm(arch):
    build.check_arch(ArmEncoder.name, arch)


@pytest.mark.parametrize("arch", ["x86_64", "amd64"])
def test_the_arm_file_fails_an_intel_build(arch):
    with pytest.raises(SystemExit, match="does not match"):
        build.check_arch(ArmEncoder.name, arch)


def test_an_unknown_arch_fails_the_build():
    with pytest.raises(SystemExit, match="unknown arch"):
        build.check_arch(ArmEncoder.name, "riscv64")


def test_main_writes_vectors_and_a_map_that_matches_them(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(build.Embedder, "default", classmethod(lambda cls, *a, **k: ArmEncoder()))
    build.main(["--cache", str(tmp_path), "--expect-arch", "aarch64", "--map-method", "pca"])
    manifest = json.loads((tmp_path / "vectors" / "manifest.json").read_text())
    assert manifest["model"] == ArmEncoder.name and manifest["rows"] == 2000
    assert (tmp_path / "vectors" / "READY").exists()
    game_map = load_map(tmp_path / "map", manifest)
    assert game_map is not None and game_map.method == "pca" and game_map.xy.shape == (2000, 2)
    out = capsys.readouterr().out
    assert "vectors 2000 x 64" in out and "map pca, 8 clusters" in out


def test_main_stops_before_building_anything_on_the_wrong_arch(tmp_path, monkeypatch):
    monkeypatch.setattr(build.Embedder, "default", classmethod(lambda cls, *a, **k: ArmEncoder()))
    with pytest.raises(SystemExit):
        build.main(["--cache", str(tmp_path), "--expect-arch", "x86_64"])
    assert not (tmp_path / "vectors").exists()
