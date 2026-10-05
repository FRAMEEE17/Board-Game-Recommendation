# board_game_reco/embedder.py
from __future__ import annotations

import os
import platform
from pathlib import Path

import numpy as np

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DIMENSIONS = 384
# Full-precision reference build in the same repo, 471 MB. Used only by the perf suite (N13).
FP32_ONNX = "onnx/model.onnx"


def threads_from_env() -> int | None:
    """BGR_THREADS caps onnxruntime's thread pool. Unset, empty or 0 keeps the onnxruntime default."""
    value = os.environ.get("BGR_THREADS", "").strip()
    return int(value) if value and int(value) > 0 else None


def _onnx_file() -> str:
    if platform.machine().lower() in ("arm64", "aarch64"):
        return "onnx/model_qint8_arm64.onnx"
    return "onnx/model_quint8_avx2.onnx"


class Embedder:
    """The official int8 ONNX build of the multilingual MiniLM, mean-pooled and L2-normalized."""

    def __init__(self, model_path: Path, tokenizer_path: Path, name: str, threads: int | None = None) -> None:
        """threads sets intra_op_num_threads. onnxruntime sizes its pool from the host's cores and ignores
        a container's CPU quota, so a 2-CPU container on a 12-core host needs threads=2."""
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.name = name
        options = ort.SessionOptions()
        if threads:
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
        self._session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
        self._inputs = {i.name for i in self._session.get_inputs()}
        self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
        self._tokenizer.enable_truncation(max_length=128)
        self._tokenizer.enable_padding()

    @classmethod
    def default(cls, onnx: str | None = None) -> Embedder:
        """The int8 build for this CPU. Pass FP32_ONNX for the full-precision reference.

        With HF_HUB_OFFLINE=1 both files resolve from the local cache and no request is made.
        """
        from huggingface_hub import hf_hub_download

        onnx = onnx or _onnx_file()
        return cls(
            Path(hf_hub_download(REPO, onnx)),
            Path(hf_hub_download(REPO, "tokenizer.json")),
            f"{REPO}/{onnx}",
            threads=threads_from_env(),
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
