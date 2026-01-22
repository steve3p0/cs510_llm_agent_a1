from __future__ import annotations

import os
import sys
from typing import List, Dict, Any, Optional

import dspy
from datasets import load_dataset

from .chroma_rm import ChromaRM


DATASET_NAME = "rag-datasets/rag-mini-bioasq"
QA_SUBSET = "question-answer-passages"


class BioASQAnswer(dspy.Signature):
    """Answer biomedical questions using retrieved passages."""
    context = dspy.InputField(desc="Relevant biomedical passages")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="A concise, accurate biomedical answer.")


class RAGBioASQ(dspy.Module):
    def __init__(self, k: int = 5):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        # ChainOfThought is fine, but it requires an LM configured.
        self.generate = dspy.ChainOfThought(BioASQAnswer)

    def forward(self, question: str):
        ctx = self.retrieve(question).passages
        pred = self.generate(context="\n\n".join(ctx), question=question)
        return dspy.Prediction(answer=pred.answer, context=ctx)


def _load_qa_dataset(split: Optional[str] = None):
    """
    Load QA subset with robust split selection.

    Some RAG datasets use 'test', others might use 'validation' or 'train'.
    Prefer: test -> validation -> train -> first available
    """
    ds_dict = load_dataset(DATASET_NAME, QA_SUBSET)  # DatasetDict
    available_splits = list(ds_dict.keys())
    if not available_splits:
        raise RuntimeError(f"No splits found for {DATASET_NAME}/{QA_SUBSET}")

    if split is not None:
        if split not in ds_dict:
            raise ValueError(f'Unknown split "{split}". Should be one of {available_splits}.')
        return ds_dict[split]

    preferred = ["test", "validation", "train"]
    chosen = next((s for s in preferred if s in ds_dict), available_splits[0])
    return ds_dict[chosen]


def load_bioasq_examples(n: Optional[int] = 50, split: Optional[str] = None) -> List[Dict[str, Any]]:
    ds = _load_qa_dataset(split=split)
    if n is not None:
        ds = ds.select(range(min(n, len(ds))))

    rows: List[Dict[str, Any]] = []
    for r in ds:
        rows.append(
            {
                "question": r.get("question", ""),
                "gold_answer": r.get("answer", ""),
                "id": r.get("id"),
            }
        )
    return rows


def run_demo_question(rag: RAGBioASQ, q: str) -> None:
    out = rag(question=q)
    print("\nQUESTION:", q)
    print("\nANSWER:", out.answer)
    print("\nTOP CONTEXT PASSAGES (truncated):")
    for i, p in enumerate(out.context[:3], start=1):
        p = p or ""
        print(f"\n[{i}] {p[:350]}{'...' if len(p) > 350 else ''}")


def _configure_dspy() -> None:
    """
    Configure DSPy RM and (optionally) an LM via environment variables.

    Required for generation:
      - OPENAI_API_KEY (if using an OpenAI LM)
    Optional:
      - DSPY_LM: e.g. "openai/gpt-4o-mini" or your preferred model string
      - CHROMA_DIR: default "data/chroma_bioasq"
      - CHROMA_COLLECTION: default "bioasq_text_corpus"
    """
    chroma_dir = os.getenv("CHROMA_DIR", "data/chroma_bioasq")
    chroma_collection = os.getenv("CHROMA_COLLECTION", "bioasq_text_corpus")

    # Configure retriever (RM)
    rm = ChromaRM(persist_dir=chroma_dir, collection_name=chroma_collection)
    dspy.settings.configure(rm=rm)

    # Configure LM if possible (recommended)
    api_key = os.getenv("OPENAI_API_KEY")
    lm_name = os.getenv("DSPY_LM", "openai/gpt-4o-mini")

    if api_key:
        # DSPy will pick up OPENAI_API_KEY from the environment in many setups,
        # but we pass it explicitly for clarity/portability.
        lm = dspy.LM(lm_name, api_key=api_key)
        dspy.settings.configure(lm=lm)
        print(f"LM configured: {lm_name}")
    else:
        # Retrieval-only mode: you can still see top passages, but generation will fail if called.
        # We'll warn clearly so it doesn't look like a bug.
        print("WARNING: OPENAI_API_KEY not set. LM not configured; generation will fail.")
        print("Set OPENAI_API_KEY (and optionally DSPY_LM) in your Run/Debug configuration.\n")


def _parse_args(argv: List[str]) -> Dict[str, Any]:
    """
    Minimal argument parsing without external deps.
    Supported:
      --k=5
      --split=test
    """
    out: Dict[str, Any] = {"k": 5, "split": None}
    for a in argv:
        if a.startswith("--k="):
            out["k"] = int(a.split("=", 1)[1])
        elif a.startswith("--split="):
            out["split"] = a.split("=", 1)[1]
    return out


def main():
    args = _parse_args(sys.argv[1:])

    # Ensure Chroma index exists early with a helpful error if not.
    try:
        _configure_dspy()
    except Exception as e:
        print("ERROR: Failed to configure retriever (Chroma).")
        print("Most common cause: you have not run the index build yet:")
        print("  python -m bioasq.build_index")
        print(f"\nDetails: {e}")
        raise

    rag = RAGBioASQ(k=args["k"])

    print("BioASQ RAG ready. Type a question (or 'exit').\n")
    while True:
        q = input("Q> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break

        # If LM is not configured, calling rag() will raise when ChainOfThought runs.
        # We handle that gracefully: show retrieved passages anyway.
        try:
            run_demo_question(rag, q)
        except Exception as e:
            print("\nERROR during generation (likely LM not configured).")
            print("Retrieved passages are still available; showing them below.\n")

            # Show retrieval-only context
            ctx = dspy.Retrieve(k=args["k"])(q).passages
            print("QUESTION:", q)
            print("\nTOP CONTEXT PASSAGES (truncated):")
            for i, p in enumerate(ctx[:3], start=1):
                p = p or ""
                print(f"\n[{i}] {p[:350]}{'...' if len(p) > 350 else ''}")
            print(f"\nDetails: {e}\n")


if __name__ == "__main__":
    main()
