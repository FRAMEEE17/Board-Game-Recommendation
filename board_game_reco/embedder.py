# board_game_reco/embedder.py
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIMENSIONS = 384


def _onnx_file() -> str:
    if platform.machine().lower() in ("arm64", "aarch64"):
        return "onnx/model_qint8_arm64.onnx"
    return "onnx/model_quint8_avx2.onnx"


class Embedder:
    """The official int8 ONNX build of the multilingual MiniLM, mean-pooled and L2-normalized."""

    def __init__(self, model_path: Path, tokenizer_path: Path, name: str) -> None:
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        self._session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=128)
        self._tokenizer.enable_padding()

    @classmethod
    def default(cls) -> Embedder:
        from huggingface_hub import hf_hub_download

        onnx = _onnx_file()
        return cls(
            Path(hf_hub_download(REPO, onnx)),
            Path(hf_hub_download(REPO, "tokenizer.json")),
            f"{REPO}/{onnx}",
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, DIMENSIONS), dtype=np.float32)
        batches = [self._encode_batch(texts[i : i + 64]) for i in range(0, len(texts), 64)]
        return np.vstack(batches).astype(np.float32)

    def _encode_batch(self, texts: list[str]) -> np.ndarray:
        encodings = self._tokenizer.encode_batch(texts)
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        feed = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in self._inputs:
            feed["token_type_ids"] = np.zeros_like(ids)
        hidden = np.asarray(self._session.run(None, feed)[0])
        pooled = (hidden * mask[..., None]).sum(axis=1) / np.clip(mask.sum(axis=1, keepdims=True), 1, None)
        return pooled / np.linalg.norm(pooled, axis=1, keepdims=True)
