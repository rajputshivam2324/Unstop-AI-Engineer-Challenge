"""<MODULE PURPOSE>
Embedder wrapper around sentence-transformers all-MiniLM-L6-v2: L2-normalized
float32 vectors (dot product in FAISS = cosine), zero-norm-safety, model
introspection (dim, max_seq_tokens), and a timed warmup.

<PUBLIC API> Embedder
<DEPENDS ON> sentence_transformers, numpy, config (EmbeddingConfig), logging.
"""

from __future__ import annotations

import gc
import logging
import time

import numpy as np

from integrity_engine.config import EmbeddingConfig

logger = logging.getLogger("integrity_engine.embeddings")

DEFAULT_MAX_SEQ_TOKENS = 256  # fallback when the model does not expose max_seq_length


class Embedder:
    """Thin, deterministic sentence-transformer encoder (all modalities -> 384-d)."""

    def __init__(self, model_name: str, cache_dir: str | None, device: str = "cpu",
                 batch_size: int = 32, normalize: bool = True) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.normalize = normalize
        self.cache_dir = cache_dir
        self._model = SentenceTransformer(model_name, device=device,
                                          cache_folder=cache_dir)
        self.warmup()

    @property
    def dim(self) -> int:
        """Embedding dimensionality (384 for all-MiniLM-L6-v2)."""
        getter = (getattr(self._model, "get_embedding_dimension", None)
                  or self._model.get_sentence_embedding_dimension)
        return int(getter())

    @property
    def max_seq_tokens(self) -> int:
        """Model context length in tokens (D3: introspect; default 256)."""
        value = getattr(self._model, "max_seq_length", None)
        return int(value) if value else DEFAULT_MAX_SEQ_TOKENS

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize,
            show_progress_bar=False,
            batch_size=self.batch_size,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def encode_corpus(self, texts: list[str]) -> np.ndarray:
        """(N, dim) float32 matrix, L2-normalized rows; zero-length input -> (0, dim).

        Zero-norm rows occur for empty texts; they are left as zeros (never
        renormalized — avoids divide-by-zero; their IP scores are 0).
        """
        matrix = self._encode(list(texts))
        if matrix.shape[0] == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        if self.normalize:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            safe = np.where(norms > 0, norms, 1.0).astype(np.float32)
            matrix = matrix / safe  # zero-norm rows stay zero
        return np.ascontiguousarray(matrix, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        """(1, dim) normalized vector for a single query text."""
        return self.encode_corpus([text])

    def encode_pairs_cosine(self, left: list[str], right: list[str]) -> list[float]:
        """Batched cosine similarity for aligned string pairs (sentence evidence)."""
        if not left or not right or len(left) != len(right):
            return []
        a = self.encode_corpus(left)
        b = self.encode_corpus(right)
        return [float(np.dot(a[i], b[i])) for i in range(len(left))]

    def warmup(self) -> None:
        """Load the model and time one dummy encode; logs milliseconds."""
        start = time.monotonic()
        self.encode_corpus(["warmup sentence for model load timing"])
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("embeddings: %s warm (%.0f ms, dim=%d, max_seq=%d)",
                    self.model_name, elapsed_ms, self.dim, self.max_seq_tokens)

    def release(self) -> None:
        """Explicitly drop the torch model (RAM discipline between stages)."""
        self._model = None  # type: ignore[assignment]
        gc.collect()


def build_embedder(cfg: EmbeddingConfig, hf_home: str) -> Embedder:
    """Factory used by the CLI; logs resolved model/device before construction."""
    logger.info("embeddings: loading %s on %s (HF_HOME=%s)",
                cfg.model_name, cfg.device, hf_home)
    return Embedder(cfg.model_name, hf_home, device=cfg.device,
                    batch_size=cfg.batch_size)
