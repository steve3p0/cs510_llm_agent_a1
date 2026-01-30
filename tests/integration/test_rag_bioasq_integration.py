# tests/integration/test_rag_bioasq_integration.py
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import chromadb
import pytest


pytestmark = pytest.mark.integration


def _make_tiny_chroma(persist_dir: Path, collection_name: str) -> None:
    """Create a tiny persistent Chroma collection for deterministic retrieval tests."""
    persist_dir.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(persist_dir))
    col = client.get_or_create_collection(name=collection_name)

    # Keep docs short and unambiguous to avoid flaky retrieval.
    docs = [
        (
            "d_linux",
            "On 32-bit Linux, low memory is permanently mapped for the kernel. "
            "High memory is not permanently mapped and needs temporary mapping.",
        ),
        (
            "d_enamel",
            "Tooth enamel is highly mineralized. Fluoride can help remineralization "
            "and reduce demineralization under acidic challenge.",
        ),
        ("d_other", "This passage is unrelated to the test questions."),
    ]

    col.add(
        ids=[d[0] for d in docs],
        documents=[d[1] for d in docs],
        metadatas=[{"source": "tiny"} for _ in docs],
    )


def _run_rag_bioasq(repo_root: Path, env: dict[str, str], stdin_text: str) -> subprocess.CompletedProcess[str]:
    """Run the interactive rag_bioasq module with provided stdin."""
    import sys

    run_env = os.environ.copy()
    run_env.update(env)

    # Ensure retrieval-only mode (no LM). Your program should still show passages.
    run_env.pop("OPENAI_API_KEY", None)

    return subprocess.run(
        [sys.executable, "-m", "bioasq.rag_bioasq", "--k=3"],
        cwd=str(repo_root),
        env=run_env,
        text=True,
        input=stdin_text,
        capture_output=True,
    )


def test_integration_retrieval_only_shows_expected_evidence(tmp_path: Path):
    """
    E2E integration test:
      - Create a tiny Chroma DB
      - Run the real interactive program
      - Ask a question via stdin
      - Assert the expected evidence appears in TOP CONTEXT PASSAGES
    """
    repo_root = Path(__file__).resolve().parents[2]  # project root
    persist_dir = tmp_path / "chroma_test"
    collection = "test_collection"

    _make_tiny_chroma(persist_dir, collection)

    proc = _run_rag_bioasq(
        repo_root=repo_root,
        env={
            "CHROMA_DIR": str(persist_dir),
            "CHROMA_COLLECTION": collection,
        },
        stdin_text="what are high memory and low memory on linux?\nexit\n",
    )

    assert proc.returncode == 0, f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"

    out = proc.stdout.lower()

    # Confirms we ran the actual interactive loop
    assert "bioasq rag ready" in out
    assert "top context passages" in out

    # Confirms retrieval got the right passage
    assert "low memory" in out
    assert "high memory" in out

    # Confirms retrieval-only fallback executed (LM not configured)
    assert "error during generation" in out


def test_integration_starts_cleanly_when_chroma_empty(tmp_path: Path):
    """
    E2E integration test:
      - Point CHROMA_DIR at a new/empty folder
      - Program should still start the REPL and not crash
    """
    repo_root = Path(__file__).resolve().parents[2]
    persist_dir = tmp_path / "chroma_empty"
    collection = "empty_collection"

    proc = _run_rag_bioasq(
        repo_root=repo_root,
        env={
            "CHROMA_DIR": str(persist_dir),
            "CHROMA_COLLECTION": collection,
        },
        stdin_text="exit\n",
    )

    # In your implementation, this should be a clean exit.
    assert proc.returncode == 0, f"STDERR:\n{proc.stderr}\nSTDOUT:\n{proc.stdout}"

    out = proc.stdout.lower()
    assert "bioasq rag ready" in out
    assert "type a question" in out
    assert "q>" in out
