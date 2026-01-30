# CS510 LLM Agents – Assignment 1  
**RAG + BioASQ Utilities with Unit Tests**

This project contains utility code and unit tests for working with a simplified
Retrieval-Augmented Generation (RAG) pipeline using the BioASQ dataset.
It focuses on **data loading, argument parsing, index-building helpers, and
ChromaDB-related logic**, with an emphasis on **correctness and testability**.

All functionality is covered by fast, deterministic unit tests using `pytest`.

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
│   └── test_rag_bioasq.py
├── requirements.txt
├── requirements-dev.txt
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

## Running Unit Tests

### Command line (recommended baseline)

From the project root:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Expected output:

```text
11 passed in <time>s
```

---

### PyCharm Run/Debug Configuration

To run all tests in PyCharm:

- **Run type:** Python tests → pytest  
- **Interpreter:** Project virtual environment  
- **Script path:**  
  ```
  C:\workspace_courses\cs510_llm_agent_a1
  ```
- **Working directory:**  
  ```
  C:\workspace_courses\cs510_llm_agent_a1
  ```

> Note: On Windows, using an explicit absolute path is more reliable than
> `<project_root>` for PyCharm’s pytest runner.

---

## Testing Philosophy

- Tests are **pure unit tests**
- No network calls
- No model loading
- No large datasets loaded at import time
- All external behavior is mocked or reduced to deterministic inputs

This ensures:
- Fast execution (< 0.1s total)
- Reliable grading
- Clear failure modes

---

## What Is Tested

### `build_index`
- Argument parsing
- Default and fallback behavior
- Passage selection logic

### `chroma_rm`
- Empty collection behavior
- Query handling
- Returned structure (scores + metadata)

### `rag_bioasq`
- Dataset loading logic
- Split selection (`passages`, `train`)
- Error handling for invalid splits
- Field extraction and normalization

---

## Notes

- `pytest.ini` is intentionally **not used**; pytest default discovery is sufficient.
- `pytest-cov` is intentionally omitted because coverage reporting is not required.
- The project is structured to prioritize clarity and correctness over performance.

---

## Author

**Steve Braich**  
CS510 – LLM Agents  
Portland State University
