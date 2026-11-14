from .base import Backend, CostMeter, AdapterScopeError
from .toy import ToyBackend


def load_backend(kind: str, path: str, **kw):
    if kind == "toy":
        return ToyBackend(path)
    if kind == "hf":
        from .hf import HFBackend   # lazy: torch/transformers are optional
        return HFBackend(path, **kw)
    raise ValueError(f"unknown backend kind: {kind}")
