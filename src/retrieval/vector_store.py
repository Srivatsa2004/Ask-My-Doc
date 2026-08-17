"""
vector_store.py
----------------
Thin wrapper around Qdrant for storing and searching chunk embeddings.

Two connection modes, chosen by what you pass in:
- Local/embedded (default): `QdrantVectorStore(path="data/processed/qdrant")`
  runs Qdrant on-disk with no server process. Good for dev, testing, and a
  single-machine portfolio deployment.
- Hosted: `QdrantVectorStore(url="https://xyz.cloud.qdrant.io", api_key="...")`
  points at a real Qdrant Cloud cluster for production use. Same interface
  either way -- calling code doesn't change.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple, Optional

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

DEFAULT_LOCAL_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "qdrant"
COLLECTION_NAME = "chunks"


class VectorHit(NamedTuple):
    chunk_id: str
    score: float


class QdrantVectorStore:
    def __init__(
        self,
        dim: int,
        path: Optional[Path] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        if url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            resolved_path = path or DEFAULT_LOCAL_PATH
            resolved_path.parent.mkdir(parents=True, exist_ok=True)
            self.client = QdrantClient(path=str(resolved_path))

        self.dim = dim
        self._ensure_collection()

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self.client.get_collections().collections]
        if COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=qmodels.VectorParams(size=self.dim, distance=qmodels.Distance.COSINE),
            )

    def upsert(self, chunk_ids: list[str], vectors: list[list[float]]) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must be the same length")
        # Qdrant point IDs must be int/UUID, not arbitrary strings -- we store the
        # real chunk_id in the payload and let Qdrant assign/accept a stable
        # integer id derived from a hash, so re-upserting the same chunk_id
        # overwrites rather than duplicates.
        points = [
            qmodels.PointStruct(
                id=_stable_int_id(cid),
                vector=vec,
                payload={"chunk_id": cid},
            )
            for cid, vec in zip(chunk_ids, vectors)
        ]
        self.client.upsert(collection_name=COLLECTION_NAME, points=points)

    def search(self, query_vector: list[float], top_k: int = 20) -> list[VectorHit]:
        results = self.client.query_points(
            collection_name=COLLECTION_NAME, query=query_vector, limit=top_k,
        ).points
        return [VectorHit(chunk_id=r.payload["chunk_id"], score=r.score) for r in results]

    def count(self) -> int:
        return self.client.count(collection_name=COLLECTION_NAME).count

    def close(self) -> None:
        self.client.close()


def _stable_int_id(chunk_id: str) -> int:
    """Deterministically map a chunk_id string to an int Qdrant point ID,
    so re-ingesting the same chunk overwrites its old vector instead of
    creating a duplicate point."""
    import hashlib
    return int(hashlib.sha256(chunk_id.encode("utf-8")).hexdigest()[:12], 16)
