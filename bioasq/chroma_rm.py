from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import chromadb


@dataclass
class _Passage:
    """
    Minimal passage object compatible with DSPy Retrieve.

    DSPy Retrieve expects objects with a `.long_text` attribute.
    Some DSPy versions do not expose dspy.Passage, so we provide our own.
    """
    long_text: str
    score: Optional[float] = None
    meta: Optional[Dict[str, Any]] = None


@dataclass
class ChromaRM:
    """
    DSPy Retrieval Model (RM) adapter backed by ChromaDB.

    Returns a list of passage objects that have `.long_text` so DSPy Retrieve works.
    """
    persist_dir: str = "data/chroma_bioasq"
    collection_name: str = "bioasq_text_corpus"

    def __post_init__(self) -> None:
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        return self._collection.count()

    def __call__(self, query: str, k: int = 5) -> List[_Passage]:
        if k <= 0:
            return []

        if self._collection.count() == 0:
            return []

        res = self._collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents", "distances", "metadatas"],
        )

        documents: List[str] = (res.get("documents") or [[]])[0]
        distances: List[float] = (res.get("distances") or [[]])[0]
        metadatas: List[Dict[str, Any]] = (res.get("metadatas") or [[]])[0]

        passages: List[_Passage] = []
        for i, doc in enumerate(documents):
            if not doc:
                continue

            # Chroma cosine distance: lower is better; convert to similarity-like score
            score: Optional[float] = None
            if i < len(distances) and isinstance(distances[i], (int, float)):
                score = 1.0 - float(distances[i])

            meta: Optional[Dict[str, Any]] = None
            if i < len(metadatas) and isinstance(metadatas[i], dict):
                meta = metadatas[i]

            passages.append(_Passage(long_text=str(doc), score=score, meta=meta))

        return passages
