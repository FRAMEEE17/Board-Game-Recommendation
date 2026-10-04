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
