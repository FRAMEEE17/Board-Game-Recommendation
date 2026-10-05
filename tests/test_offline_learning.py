"""Learning test: the Docker image runs with HF_HUB_OFFLINE=1 and no network. These prove that
hf_hub_download resolves both model files from the local cache in that mode and makes no request."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def offline(code: str) -> subprocess.CompletedProcess:
    # huggingface_hub reads HF_HUB_OFFLINE once, at import, so the check runs in a fresh interpreter.
    env = os.environ | {"HF_HUB_OFFLINE": "1"}
    return subprocess.run([sys.executable, "-c", code], env=env, cwd=ROOT, capture_output=True, text=True,
                          timeout=120)


@pytest.mark.model
def test_embedder_loads_from_the_cache_with_the_hub_offline():
    from board_game_reco.embedder import Embedder, _onnx_file

    Embedder.default()  # the only step allowed to download, so the cache is filled
    done = offline("from board_game_reco.embedder import Embedder; print(Embedder.default().name)")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip().endswith(_onnx_file())


@pytest.mark.model
def test_a_file_missing_from_the_cache_fails_fast_offline():
    done = offline(
        "from huggingface_hub import hf_hub_download\n"
        "from board_game_reco.embedder import REPO\n"
        "hf_hub_download(REPO, 'onnx/no-such-file.onnx')\n"
    )
    assert done.returncode != 0
    assert "offline" in done.stderr.lower() or "LocalEntryNotFoundError" in done.stderr
