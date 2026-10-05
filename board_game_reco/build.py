# board_game_reco/build.py
"""python -m board_game_reco.build --cache DIR --expect-arch ARCH [--map-method umap|pca]

Build-time warm-up for the Docker image: downloads the int8 model and tokenizer for this CPU,
builds the catalog vectors, checks the model matches the target arch, and builds the game map.
The runtime stage only reads what this writes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .catalog import Catalog
from .embedder import Embedder
from .gamemap import METHODS, build_map
from .intent import IntentClassifier

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "boardgames.csv"
EXPECTED_ONNX = {
    "aarch64": "model_qint8_arm64.onnx",
    "arm64": "model_qint8_arm64.onnx",
    "x86_64": "model_quint8_avx2.onnx",
    "amd64": "model_quint8_avx2.onnx",
}


def check_arch(model_name: str, arch: str) -> None:
    expected = EXPECTED_ONNX.get(arch.lower())
    if expected is None:
        raise SystemExit(f"unknown arch {arch!r}. Expected one of {', '.join(EXPECTED_ONNX)}")
    if not model_name.endswith(expected):
        raise SystemExit(f"model {model_name} does not match arch {arch}: expected {expected}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="board_game_reco.build")
    parser.add_argument("--cache", type=Path, default=ROOT / ".cache", help="writes vectors/ and map/ here")
    parser.add_argument("--expect-arch", required=True, help="uname -m of the target, e.g. aarch64 or x86_64")
    parser.add_argument("--map-method", choices=METHODS, default="umap")
    args = parser.parse_args(argv)

    encoder = Embedder.default()
    check_arch(encoder.name, args.expect_arch)
    catalog = Catalog.load(CSV, args.cache / "vectors", encoder)
    IntentClassifier(encoder)  # fails the build now if the prototypes cannot be encoded
    manifest = json.loads((args.cache / "vectors" / "manifest.json").read_text())
    vectors = np.load(args.cache / "vectors" / "vectors.npy")
    game_map = build_map(vectors, catalog.games, args.cache / "map", manifest, method=args.map_method)
    digest = hashlib.sha256(vectors.tobytes()).hexdigest()[:12]
    print(f"model {encoder.name}")
    print(f"vectors {vectors.shape[0]} x {vectors.shape[1]} sha256 {digest}")
    print(f"map {game_map.method}, {len(game_map.labels)} clusters")


if __name__ == "__main__":
    main()
