"""Backend interface. A backend wraps one *bundle* (weights + tokenizer + template + config + runtime)."""
from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
import numpy as np


class AdapterScopeError(RuntimeError):
    """Raised when a bundle declares adapter files but the backend cannot establish, without
    guessing, which base model they apply to (or cannot apply them at all).

    Per project rule: an adapter must never be silently ignored. Either it is actually
    applied by the backend (adapter_status == 'applied'), or loading refuses outright
    (this exception) when there is no base model to attach it to, or the backend loads
    the base only and flags adapter_status == 'unsupported' when it can identify a base
    but cannot apply the adapter (e.g. peft not installed).
    """


@dataclass
class CostMeter:
    forward_passes: int = 0
    tokens_in: int = 0
    tokens_generated: int = 0
    seconds: float = 0.0

    def add(self, passes=0, tin=0, tgen=0, sec=0.0):
        self.forward_passes += passes
        self.tokens_in += tin
        self.tokens_generated += tgen
        self.seconds += sec

    def snapshot(self) -> dict:
        return dict(forward_passes=self.forward_passes, tokens_in=self.tokens_in,
                    tokens_generated=self.tokens_generated, seconds=round(self.seconds, 6))

    def __sub__(self, other: "CostMeter") -> "CostMeter":
        return CostMeter(self.forward_passes - other.forward_passes, self.tokens_in - other.tokens_in,
                         self.tokens_generated - other.tokens_generated, self.seconds - other.seconds)


@runtime_checkable
class Backend(Protocol):
    name: str
    cost: CostMeter
    vocab_size: int
    # adapter_status: 'none' (no adapter files in bundle) | 'applied' (adapter files present
    # and actually in the forward pass) | 'unsupported' (adapter files present but NOT applied;
    # the backend is evaluating the base model only). Never silently 'none' when files exist.
    adapter_status: str
    adapter_detail: dict

    def render(self, messages: list[dict]) -> str:
        """Apply THIS bundle's own chat template (add generation prompt)."""

    def logprobs(self, text: str) -> np.ndarray:
        """Next-token log-probabilities (float64, normalized) for already-rendered text."""

    def generate(self, text: str, max_new_tokens: int = 32) -> str:
        """Greedy continuation of already-rendered text."""

    def encode(self, text: str) -> list[int]:
        """Token ids of rendered text under THIS bundle's tokenizer."""

    def vocab_sample(self, rng: np.random.Generator, n: int) -> list[str]:
        """Printable token strings usable by prompt mutation."""


class Timer:
    def __init__(self, meter: CostMeter):
        self.m = meter

    def __enter__(self):
        self.t = time.perf_counter()
        return self

    def __exit__(self, *a):
        self.m.seconds += time.perf_counter() - self.t
