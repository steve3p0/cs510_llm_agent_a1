from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DATASET_NAME = "rag-datasets/rag-mini-bioasq"
CORPUS_SUBSET = "text-corpus"
COLLECTION_NAME = "bioasq_text_corpus"


def _get_text_field(example: Dict[str, Any]) -> str:
    """
    The corpus rows vary across RAG datasets; prefer common keys.
    """
    for key in ("text", "passage", "contents", "content", "document", "chunk", "title"):
        if key in example and example[key]:
            return str(example[key])

    # Fall back: stringify the row (last resort).
    # This is intentionally conservative: we only include non-null values.
    parts: List[str] = []
    for v in example.values():
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return " ".join(parts)


def _get_passage_id(example: Dict[str, Any], fallback_index: int) -> str:
    """
    Pick a stable passage ID if present; otherwise fall back to row index.
    """
    for k in ("passage_id", "pid", "doc_id", "id"):
        if k in example and example[k] is not None:
            return str(example[k])
    return str(fallback_index)


def _load_corpus_dataset(limit: Optional[int] = None):
    """
    Load the BioASQ text corpus subset and choose the correct split.
    The error you hit indicates the split is named 'passages' (not 'train').
    We auto-detect to keep this robust.
    """
    print(f"Loading corpus subset: {DATASET_NAME} / {CORPUS_SUBSET}")

    ds_dict = load_dataset(DATASET_NAME, CORPUS_SUBSET)  # DatasetDict
    available_splits = list(ds_dict.keys())
    if not available_splits:
        raise RuntimeError(f"No splits found for {DATASET_NAME}/{CORPUS_SUBSET}")

    preferred = ["passages", "train", "corpus"]
    split = next((s for s in preferred if s in ds_dict), available_splits[0])

    print(f"Using split '{split}'. Available splits: {available_splits}")
    ds = ds_dict[split]

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    return ds


def build_bioasq_chroma_index(
    persist_dir: str = "data/chroma_bioasq",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 256,
    limit: Optional[int] = None,
) -> None:
    """
    Build a persistent ChromaDB collection for rag-mini-bioasq's text corpus.

    - persist_dir: path where Chroma stores its persistent data
    - model_name: SentenceTransformers model to embed passages
    - batch_size: how many passages to embed/add per batch
    - limit: optional cap for quick smoke tests (e.g., 2000)
    """
    os.makedirs(persist_dir, exist_ok=True)

    # Load dataset (robust split selection)
    ds = _load_corpus_dataset(limit=limit)

    # Create Chroma persistent client + collection
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # If the collection already has data, skip rebuild (simple guard).
    existing = collection.count()
    if existing > 0:
        print(f"Chroma collection '{COLLECTION_NAME}' already has {existing} items. Skipping rebuild.")
        return

    print(f"Embedding model: {model_name}")
    embedder = SentenceTransformer(model_name)

    ids: List[str] = []
    docs: List[str] = []
    metas: List[Dict[str, Any]] = []

    def flush() -> None:
        if not ids:
            return
        # Normalize embeddings for cosine similarity
        embeddings = embedder.encode(
            docs,
            show_progress_bar=False,
            normalize_embeddings=True,
        ).tolist()
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
        ids.clear()
        docs.clear()
        metas.clear()

    total = len(ds)
    print(f"Indexing {total:,} corpus passages into Chroma at '{persist_dir}' ...")

    for i, row in enumerate(tqdm(ds, total=total)):
        pid = _get_passage_id(row, fallback_index=i)
        text = _get_text_field(row)

        if not text.strip():
            continue

        ids.append(pid)
        docs.append(text)
        metas.append(
            {
                "source": "rag-mini-bioasq",
                "subset": CORPUS_SUBSET,
                "split_hint": "passages",
                "row_index": i,
            }
        )

        if len(ids) >= batch_size:
            flush()

    flush()
    print(f"Done. Final collection count: {collection.count():,}")


if __name__ == "__main__":
    # For a quick smoke test, set limit=2000 and verify it finishes,
    # then remove the limit for full indexing.
    build_bioasq_chroma_index()
