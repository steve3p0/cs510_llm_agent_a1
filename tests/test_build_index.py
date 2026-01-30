import types
import pytest

from tests.test_utils import import_with_stubs

def _fake_datasets(ds_dict):
    mod = types.SimpleNamespace()
    def load_dataset(name, subset):
        return ds_dict
    mod.load_dataset = load_dataset
    return mod

def test_get_text_field_prefers_common_keys():
    # Stub heavy deps so module import is cheap
    stubs = {
        "chromadb": types.SimpleNamespace(),
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=object),
        "tqdm": types.SimpleNamespace(tqdm=lambda x, total=None: x),
        "datasets": _fake_datasets({"passages": []}),
    }
    m = import_with_stubs("bioasq.build_index", stubs)

    assert m._get_text_field({"text": "hello"}) == "hello"
    assert m._get_text_field({"passage": "p"}) == "p"
    assert m._get_text_field({"contents": "c"}) == "c"

def test_get_text_field_fallback_stringifies_non_null_values():
    stubs = {
        "chromadb": types.SimpleNamespace(),
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=object),
        "tqdm": types.SimpleNamespace(tqdm=lambda x, total=None: x),
        "datasets": _fake_datasets({"passages": []}),
    }
    m = import_with_stubs("bioasq.build_index", stubs)

    out = m._get_text_field({"a": None, "b": " hi ", "c": 3})
    assert out == "hi 3"

def test_get_passage_id_prefers_known_keys_else_fallback():
    stubs = {
        "chromadb": types.SimpleNamespace(),
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=object),
        "tqdm": types.SimpleNamespace(tqdm=lambda x, total=None: x),
        "datasets": _fake_datasets({"passages": []}),
    }
    m = import_with_stubs("bioasq.build_index", stubs)

    assert m._get_passage_id({"passage_id": "p1"}, 99) == "p1"
    assert m._get_passage_id({"doc_id": 123}, 99) == "123"
    assert m._get_passage_id({}, 99) == "99"

def test_load_corpus_dataset_prefers_passages_split():
    # Fake HF Dataset with select + __len__
    class FakeDS(list):
        def select(self, idxs):
            return FakeDS([self[i] for i in idxs])

    fake_passages = FakeDS([{"text":"t1"},{"text":"t2"},{"text":"t3"}])
    ds_dict = {"passages": fake_passages, "train": FakeDS([{"text":"x"}])}

    stubs = {
        "chromadb": types.SimpleNamespace(),
        "sentence_transformers": types.SimpleNamespace(SentenceTransformer=object),
        "tqdm": types.SimpleNamespace(tqdm=lambda x, total=None: x),
        "datasets": _fake_datasets(ds_dict),
    }
    m = import_with_stubs("bioasq.build_index", stubs)

    ds = m._load_corpus_dataset(limit=2)
    assert len(ds) == 2
    assert ds[0]["text"] == "t1"
