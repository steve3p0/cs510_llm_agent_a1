import types

from tests.test_utils import import_with_stubs

def test_chromarm_empty_k_returns_empty_list():
    # stub chromadb client/collection
    class FakeCollection:
        def count(self): return 10
        def query(self, **kwargs): return {}
    class FakeClient:
        def __init__(self, path): pass
        def get_or_create_collection(self, name, metadata=None): return FakeCollection()
    chromadb = types.SimpleNamespace(PersistentClient=FakeClient)

    m = import_with_stubs("bioasq.chroma_rm", {"chromadb": chromadb})

    rm = m.ChromaRM(persist_dir="x", collection_name="y")
    assert rm("q", k=0) == []

def test_chromarm_empty_collection_returns_empty_list():
    class FakeCollection:
        def count(self): return 0
        def query(self, **kwargs): return {}
    class FakeClient:
        def __init__(self, path): pass
        def get_or_create_collection(self, name, metadata=None): return FakeCollection()
    chromadb = types.SimpleNamespace(PersistentClient=FakeClient)

    m = import_with_stubs("bioasq.chroma_rm", {"chromadb": chromadb})

    rm = m.ChromaRM(persist_dir="x", collection_name="y")
    assert rm("q", k=5) == []

def test_chromarm_query_builds_passages_with_score_and_meta():
    class FakeCollection:
        def count(self): return 2
        def query(self, **kwargs):
            return {
                "documents": [["doc1", "doc2"]],
                "distances": [[0.2, 0.7]],
                "metadatas": [[{"a":1}, {"b":2}]],
            }
    class FakeClient:
        def __init__(self, path): pass
        def get_or_create_collection(self, name, metadata=None): return FakeCollection()
    chromadb = types.SimpleNamespace(PersistentClient=FakeClient)

    m = import_with_stubs("bioasq.chroma_rm", {"chromadb": chromadb})
    rm = m.ChromaRM(persist_dir="x", collection_name="y")
    passages = rm("q", k=5)

    assert len(passages) == 2
    assert passages[0].long_text == "doc1"
    assert abs(passages[0].score - 0.8) < 1e-9
    assert passages[0].meta == {"a":1}
