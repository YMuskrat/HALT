"""Optional pinned sentence encoder; loading happens only on explicit construction."""

from __future__ import annotations

import re
from typing import Any

from halt.types import StepEmbeddings

MINILM_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MINILM_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"


class SentenceTransformerEmbeddings:
    """L2-normalized embeddings with cosine similarity through inner product.

    The encoder's native max sequence length and truncation behavior are
    recorded. REFRAIN source fidelity is a method-level claim, not implied by
    selecting this provider.
    """

    def __init__(
        self,
        model_id: str = MINILM_MODEL_ID,
        *,
        revision: str | None = None,
        device: str = "cpu",
        cache_folder: str | None = None,
        local_files_only: bool = False,
    ) -> None:
        revision = revision or (MINILM_REVISION if model_id == MINILM_MODEL_ID else None)
        if revision is None or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
            raise ValueError("Embedding model requires an immutable 40-character commit revision")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install halt-reasoning[embeddings] to use embeddings") from exc
        self.model_id = model_id
        self.revision = revision
        self.model: Any = SentenceTransformer(
            model_id,
            revision=revision,
            device=device,
            cache_folder=cache_folder,
            local_files_only=local_files_only,
        )

    @property
    def provenance(self) -> dict[str, Any]:
        return {
            "encoder": self.model_id,
            "revision": self.revision,
            "normalization": "L2",
            "similarity": "cosine_dot_product",
            "max_seq_length": self.model.max_seq_length,
            "truncation": "encoder_native",
        }

    def input_tokens(self, steps: tuple[str, ...]) -> int:
        if not steps:
            return 0
        encoded = self.model.tokenize(list(steps))
        return int(encoded["attention_mask"].sum().item())

    def embed(self, steps: tuple[str, ...]) -> StepEmbeddings:
        if not steps or any(not step.strip() for step in steps):
            raise ValueError("Embedding steps must be nonempty text")
        vectors = self.model.encode(
            list(steps), normalize_embeddings=True, show_progress_bar=False,
            convert_to_numpy=True,
        )
        return StepEmbeddings(
            tuple(tuple(float(x) for x in vector) for vector in vectors),
            self.model_id,
            self.revision,
        )
