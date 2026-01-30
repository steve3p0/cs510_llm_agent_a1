# CS510 LLM Agents – Assignment 1  
**Retrieval-Augmented Generation (RAG) with BioASQ**

This repository contains the solution for **Assignment 1** of **CS 510: LLM Agents** (Portland State University, Dr. Suresh Singh).

The project implements a simplified **Retrieval-Augmented Generation (RAG)** pipeline over the **BioASQ** corpus using **ChromaDB**, with a clear separation between:

- **Retrieval** (fully offline, deterministic)
- **Generation** (optional, LLM-backed)

The codebase emphasizes **correctness, transparency, and testability**, and includes both **unit tests** and **end-to-end integration tests**.

---

## Project Structure

```text
cs510_llm_agent_a1/
├── bioasq/
│   ├── __init__.py
│   ├── build_index.py
│   ├── chroma_rm.py
│   └── rag_bioasq.py
├── tests/
│   ├── test_build_index.py
│   ├── test_chroma_rm.py
│   ├── test_rag_bioasq.py
│   └── integration/
│       └── test_rag_bioasq_integration.py
├── main.py
├── requirements.txt
├── requirements-dev.txt
├── pytest.ini
└── README.md
```

---

## Environment Setup

### 1. Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

Runtime dependencies:

```powershell
pip install -r requirements.txt
```

Development / testing dependencies:

```powershell
pip install -r requirements-dev.txt
```

---

## Optional LLM Configuration (`.env.local`)

LLM-based answer generation is **optional**.  
Retrieval works without any API keys.

If you want to enable generation using OpenAI:

### Create a local environment file (not committed)

Create a file named `.env.local` in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

⚠️ **Do not commit this file to GitHub.**  
It is intentionally excluded via `.gitignore`.

If no API key is present, the system will:
- build / load the BioASQ index
- retrieve relevant passages
- display retrieved context
- **skip generation gracefully** with a warning

This behavior is intentional and covered by integration tests.

---

## Running the Application

### Start the interactive RAG system

From the project root:

```powershell
python main.py
```

Expected startup output (abridged):

```text
BioASQ RAG ready. Type a question (or 'exit').
Q>
```

You can then type questions interactively:

```text
Q> What is the meaning of life?
```

If no LLM is configured, retrieved passages will still be shown.

Exit the program with:

```text
Q> exit
```

---

## Running Unit Tests

Unit tests are **fast, deterministic, and offline**.

From the project root:

```powershell
pytest -q
```

Expected result:

```text
11 passed in <time>s
```

---

## Running Integration Tests

Integration tests validate **end-to-end system behavior**, including:

- persistent ChromaDB usage
- interactive input/output
- retrieval-only fallback when no LLM is configured

Run integration tests only:

```powershell
pytest -q -m integration
```

Expected result:

```text
2 passed, <n> deselected
```

---

## Testing Philosophy

- **No network calls**
- **No API keys required**
- **No large datasets downloaded during tests**
- Retrieval and generation are tested as **separate concerns**

Integration tests use:
- a tiny, synthetic Chroma collection
- real subprocess execution
- stdin-driven interaction

This ensures:
- reproducibility
- fast grading
- clear failure modes

---

## What Is Tested

### Unit Tests
- Argument parsing and defaults
- Chroma retrieval logic
- Dataset split handling
- Error and edge-case behavior

### Integration Tests
- End-to-end retrieval pipeline
- Interactive REPL behavior
- Retrieval-only fallback (no LLM)
- Graceful startup with empty or missing indices

---

## Notes

- API keys are **never required** for testing or grading
- Integration tests do **not** rely on `.env.local`
- `pytest.ini` is included to register custom test markers

---

## Author

**Steve Braich**  
CS 510 – LLM Agents  
Portland State University
