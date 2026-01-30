import sys
import types
import importlib

class DummySettings:
    def __init__(self):
        self.configured = {}
    def configure(self, **kwargs):
        self.configured.update(kwargs)

class DummyDspy(types.SimpleNamespace):
    def __init__(self):
        super().__init__()
        self.settings = DummySettings()

        class Retrieve:
            def __init__(self, k=5):
                self.k = k
            def __call__(self, q):
                return types.SimpleNamespace(passages=[f"passage about {q} #{i+1}" for i in range(self.k)])
        self.Retrieve = Retrieve

        class ChainOfThought:
            def __init__(self, sig):
                self.sig = sig
            def __call__(self, context, question):
                return types.SimpleNamespace(answer=f"dummy answer to: {question}")
        self.ChainOfThought = ChainOfThought

        class LM:
            def __init__(self, name, api_key=None):
                self.name=name
                self.api_key=api_key
        self.LM = LM

        class Signature: ...
        class InputField:
            def __init__(self, desc=None): self.desc=desc
        class OutputField:
            def __init__(self, desc=None): self.desc=desc
        class Module:
            def __init__(self): pass
        class Prediction(dict):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                for k,v in kwargs.items():
                    setattr(self,k,v)
        self.Signature=Signature
        self.InputField=InputField
        self.OutputField=OutputField
        self.Module=Module
        self.Prediction=Prediction

def import_with_stubs(module_name: str, stubs: dict):
    """Import a module with sys.modules preloaded with stub modules."""
    restore = {}
    for k, v in stubs.items():
        restore[k] = sys.modules.get(k)
        sys.modules[k] = v
    try:
        if module_name in sys.modules:
            del sys.modules[module_name]
        return importlib.import_module(module_name)
    finally:
        for k, old in restore.items():
            if old is None:
                del sys.modules[k]
            else:
                sys.modules[k] = old
