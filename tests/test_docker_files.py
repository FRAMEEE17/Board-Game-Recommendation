"""Checks on the Docker files that run without Docker. The image itself is checked by tests/test_image.py."""
import os
import subprocess

import pytest

from tests.conftest import ROOT

if os.environ.get("BGR_IMAGE") == "1":
    pytest.skip("checks the repo files, which the image does not carry", allow_module_level=True)

DOCKERFILE = (ROOT / "Dockerfile").read_text()
IGNORED = (ROOT / ".dockerignore").read_text().split()


def runtime_stage() -> str:
    return DOCKERFILE.split("AS runtime", 1)[1]


def test_no_env_file_reaches_the_build_context():
    assert ".env" in IGNORED and ".env.*" in IGNORED


def test_the_build_fails_if_an_env_file_is_in_the_image():
    assert "test ! -e /app/.env" in runtime_stage()


def test_no_key_is_passed_as_a_build_argument_or_baked_variable():
    assert "LLM_API_KEY" not in DOCKERFILE


def test_the_runtime_is_offline_non_root_and_gets_its_venv_from_deps():
    runtime = runtime_stage()
    assert "HF_HUB_OFFLINE=1" in runtime and "USER app" in runtime and "--uid 10001" in runtime
    assert "COPY --from=deps   /app/.venv" in runtime
    assert "--from=assets /app/.venv" not in runtime
    assert "--group map" not in DOCKERFILE.split("AS assets", 1)[0]


def test_entrypoint_passes_unknown_modes_through_as_commands():
    done = subprocess.run(["sh", str(ROOT / "docker" / "entrypoint.sh"), "echo", "passed"],
                          capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "passed"
