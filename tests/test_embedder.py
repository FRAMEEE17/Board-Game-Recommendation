# tests/test_embedder.py
import numpy as np
import pytest

from board_game_reco.embedder import Embedder


@pytest.mark.model
def test_encode_returns_unit_vectors_and_ranks_paraphrases_close():
    encoder = Embedder.default()
    vectors = encoder.encode([
        "a cooperative game for two players",
        "เกมร่วมมือสำหรับสองคน",
        "a heavy economic game about trains",
    ])
    assert vectors.shape == (3, 384)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-4)
    assert vectors[0] @ vectors[1] > vectors[0] @ vectors[2]


def test_encode_empty_list_returns_empty_matrix():
    encoder = Embedder.__new__(Embedder)
    assert encoder.encode([]).shape == (0, 384)


def test_default_can_load_the_fp32_build(monkeypatch):
    import huggingface_hub

    from board_game_reco import embedder

    asked = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: asked.append(name) or f"/tmp/{name}")
    monkeypatch.setattr(embedder.Embedder, "__init__",
                        lambda self, model, tokenizer, name, threads=None: setattr(self, "name", name))
    built = embedder.Embedder.default(embedder.FP32_ONNX)
    assert asked[0] == "onnx/model.onnx"
    assert built.name.endswith("onnx/model.onnx")


@pytest.mark.parametrize("value, expected", [(None, None), ("", None), ("0", None), ("2", 2), (" 4 ", 4)])
def test_threads_come_from_bgr_threads(monkeypatch, value, expected):
    from board_game_reco.embedder import threads_from_env

    if value is None:
        monkeypatch.delenv("BGR_THREADS", raising=False)
    else:
        monkeypatch.setenv("BGR_THREADS", value)
    assert threads_from_env() == expected


def test_default_passes_the_thread_cap_to_the_session(monkeypatch):
    import huggingface_hub

    from board_game_reco import embedder

    seen = {}
    monkeypatch.setenv("BGR_THREADS", "2")
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda repo, name: f"/tmp/{name}")
    monkeypatch.setattr(embedder.Embedder, "__init__",
                        lambda self, model, tokenizer, name, threads=None: seen.update(threads=threads))
    embedder.Embedder.default()
    assert seen == {"threads": 2}


@pytest.mark.model
def test_the_real_session_uses_the_thread_cap(monkeypatch):
    monkeypatch.setenv("BGR_THREADS", "2")
    encoder = Embedder.default()
    options = encoder._session.get_session_options()
    assert options.intra_op_num_threads == 2 and options.inter_op_num_threads == 1
    assert encoder.encode(["a quick card game"]).shape == (1, 384)
