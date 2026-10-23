"""Local dense embedding models for the retrieval stack.

The default model is ``sentence-transformers/all-MiniLM-L6-v2`` (384 dimensions)
running on CPU. ``sentence_transformers`` is imported lazily, inside the
embedder, so importing unrelated project modules never triggers a model
download or a Torch import. One embedder instance keeps the loaded model for its
lifetime; callers construct it once and reuse it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from supplier_risk.retrieval.errors import EmbeddingModelError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

DEFAULT_EMBEDDING_MODEL: Final = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DEVICE: Final = "cpu"
DEFAULT_BATCH_SIZE: Final = 32


@runtime_checkable
class Embedder(Protocol):
    """Minimal interface used to build indexes and embed queries.

    An implementation may also expose ``declared_dimension: int`` when it can
    report its vector width without loading the model. The retriever validates
    that value, together with ``model_name``, against the persisted index when
    an embedder is injected; implementations such as
    :class:`SentenceTransformerEmbedder` omit it because ``dimension`` loads the
    model on first access, and their vectors are still checked at query time.
    """

    @property
    def model_name(self) -> str:
        """Identifier of the underlying model, persisted with the index metadata."""
        ...

    @property
    def dimension(self) -> int:
        """Number of components in every embedding vector."""
        ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed one vector per input text, in input order."""
        ...

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed a single query string."""
        ...


class SentenceTransformerEmbedder:
    """CPU sentence-transformer embedder with lazy model loading."""

    def __init__(
        self,
        model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        device: str = DEFAULT_DEVICE,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        name = model_name.strip()
        if not name:
            raise EmbeddingModelError("embedding model name must not be blank")
        if batch_size < 1:
            raise EmbeddingModelError("embedding batch size must be a positive integer")
        self._model_name = name
        self._device = device
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        """Return the model's embedding size, loading the model if necessary."""
        model = self._load()
        lookup = getattr(model, "get_embedding_dimension", None)
        value = lookup() if callable(lookup) else model.get_sentence_embedding_dimension()
        if not isinstance(value, int) or value <= 0:
            raise EmbeddingModelError(
                f"embedding model {self._model_name!r} reported an invalid dimension"
            )
        return value

    def _load(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import SentenceTransformer as ModelClass
        except ImportError:
            raise EmbeddingModelError(
                "sentence-transformers is not installed; run `uv sync --locked --group dev`."
            ) from None

        try:
            model = ModelClass(self._model_name, device=self._device)
        except Exception as error:  # noqa: BLE001 - third-party load boundary
            # Any failure here (missing weights, network, bad config) is reported
            # as a project error; the exception type is kept but not its message.
            raise EmbeddingModelError(
                f"failed to load embedding model {self._model_name!r} on {self._device!r}: "
                f"{type(error).__name__}. Check local model access and try again."
            ) from None

        self._model = model
        return model

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Embed each text with the local model; vectors are L2-normalized."""
        if not texts:
            raise EmbeddingModelError("cannot embed an empty list of texts")
        model = self._load()
        try:
            vectors = model.encode(
                list(texts),
                batch_size=self._batch_size,
                device=self._device,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
        except Exception as error:  # noqa: BLE001 - third-party inference boundary
            raise EmbeddingModelError(
                f"embedding model {self._model_name!r} failed during inference: "
                f"{type(error).__name__}"
            ) from None

        array = np.asarray(vectors, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != len(texts):
            raise EmbeddingModelError(
                f"embedding model {self._model_name!r} returned an unexpected batch shape"
            )
        return np.ascontiguousarray(array)

    def embed_query(self, text: str) -> NDArray[np.float32]:
        """Embed one query string with the same model."""
        return self.embed_documents([text])[0]
