import types
import pytest

from tests.test_utils import DummyDspy, import_with_stubs

def _fake_datasets(ds_dict):
    mod = types.SimpleNamespace()
    def load_dataset(name, subset):
        return ds_dict
    mod.load_dataset = load_dataset
    return mod

def test_parse_args_defaults_and_values():
    dspy = DummyDspy()
    datasets = _fake_datasets({"test": []})
    m = import_with_stubs("bioasq.rag_bioasq", {"dspy": dspy, "datasets": datasets})

    assert m._parse_args([]) == {"k": 5, "split": None}
    assert m._parse_args(["--k=7", "--split=validation"]) == {"k": 7, "split": "validation"}

def test_load_qa_dataset_prefers_test_then_validation_then_train():
    dspy = DummyDspy()
    ds_dict = {"validation": [1], "train": [2], "test": [3]}
    datasets = _fake_datasets(ds_dict)
    m = import_with_stubs("bioasq.rag_bioasq", {"dspy": dspy, "datasets": datasets})

    assert m._load_qa_dataset(split=None) == ds_dict["test"]

    ds_dict2 = {"validation": [1], "train": [2]}
    m2 = import_with_stubs("bioasq.rag_bioasq", {"dspy": dspy, "datasets": _fake_datasets(ds_dict2)})
    assert m2._load_qa_dataset(split=None) == ds_dict2["validation"]

    ds_dict3 = {"train": [2]}
    m3 = import_with_stubs("bioasq.rag_bioasq", {"dspy": dspy, "datasets": _fake_datasets(ds_dict3)})
    assert m3._load_qa_dataset(split=None) == ds_dict3["train"]

def test_load_qa_dataset_unknown_split_raises():
    dspy = DummyDspy()
    datasets = _fake_datasets({"test": []})
    m = import_with_stubs("bioasq.rag_bioasq", {"dspy": dspy, "datasets": datasets})

    with pytest.raises(ValueError):
        m._load_qa_dataset(split="nope")

def test_load_bioasq_examples_extracts_fields_and_respects_n():
    dspy = DummyDspy()

    # mimic HF Dataset with select + iteration
    class FakeDS(list):
        def select(self, idxs):
            return FakeDS([self[i] for i in idxs])

    fake_split = FakeDS([
        {"question": "q1", "answer": "a1", "id": "1"},
        {"question": "q2", "answer": "a2", "id": "2"},
        {"question": "q3", "answer": "a3", "id": "3"},
    ])
    datasets = _fake_datasets({"test": fake_split})
    m = import_with_stubs("bioasq.rag_bioasq", {"dspy": dspy, "datasets": datasets})

    ex = m.load_bioasq_examples(n=2, split="test")
    assert len(ex) == 2
    assert ex[0]["question"] == "q1"
    assert ex[0]["gold_answer"] == "a1"
    assert ex[0]["id"] == "1"
