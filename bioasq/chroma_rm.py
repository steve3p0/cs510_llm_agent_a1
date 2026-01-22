from __future__ import annotations

from dataclasses import dataclass
from typing import List

import chromadb


@dataclass
class ChromaRM:
    """
    Minimal Retrieval Model (RM) adapter for DSPy using ChromaDB.

    Key behavior:
    - Uses a persistent Chroma client.
    - Uses get_or_create_collection so the RAG runner does NOT crash if the collection
      hasn't been created yet.
    - If the collection exists but is empty, retrieval returns an empty list (and the caller
      can emit a helpful warning).
    """
    persist_dir: str = "data/chroma_bioasq"
    collection_name: str = "bioasq_text_corpus"

    def __post_init__(self) -> None:
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        # IMPORTANT: do not use get_collection() here; it throws if missing.
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        """Return number of items in the collection."""
        return self._collection.count()

    def __call__(self, query: str, k: int = 5) -> List[str]:
        if k <= 0:
            return []

        # If the collection is empty, just return nothing.
        if self._collection.count() == 0:
            return []

        res = self._collection.query(
            query_texts=[query],
            n_results=k,
            include=["documents"],
        )
        docs = res.get("documents", [[]])[0]
        return [d for d in docs if d]
