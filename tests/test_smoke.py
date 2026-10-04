import board_game_reco


def test_package_imports():
    assert hasattr(board_game_reco, "recommend")
