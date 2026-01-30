# main.py (pseudo-integration snippet)

from bioasq.build_index import build_bioasq_chroma_index
from bioasq.rag_bioasq import main as bioasq_main

def run_bioasq():
    build_bioasq_chroma_index(persist_dir="data/chroma_bioasq")
    bioasq_main()

if __name__ == "__main__":
    run_bioasq()
