"""Paired oracle and budgeted probe-search strategies (benchmark / random / guided).

All strategies receive exactly the same budget measured in unique paired evaluations
(one reference + one derived forward pass each) and are compared at matched cost.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from . import divergence as dv
from .probes import Prompt, Mutator, make_variants, is_valid, ood_penalty


@dataclass
class PairedRecord:
    prompt: Prompt
    ref_text: str
    der_text: str
    metrics: dict
    step: int
    valid: bool
    objective: float

    @property
    def jsd(self) -> float:
        return self.metrics["jsd"]


HEARTBEAT = {"fn": None}      # optional zero-argument callback invoked after every real paired evaluation (progress display)


class PairedOracle:
    """Evaluates a prompt on reference and derived bundles. Caches by prompt content."""

    def __init__(self, ref, der, template_mode: str = "own", tau: float = 1.0,
                 lam=(1.0, 0.5, 1.0), jsd_cap: float = 20.0):
        assert template_mode in ("own", "reference")
        self.ref, self.der, self.mode = ref, der, template_mode
        self.tau, self.lam, self.cap = max(tau, 1e-9), lam, jsd_cap
        self.cache: dict[str, PairedRecord] = {}
        self.n_evals = 0

    def _texts(self, p: Prompt):
        rt = self.ref.render(p.messages)
        dt = self.der.render(p.messages) if self.mode == "own" else rt
        return rt, dt

    def evaluate(self, p: Prompt, step: int | None = None) -> PairedRecord:
        k = p.key()
        if k in self.cache:
            return self.cache[k]
        rt, dt = self._texts(p)
        m = dv.paired_metrics(self.ref.logprobs(rt), self.der.logprobs(dt))
        valid = is_valid(p)
        A = min(float(np.log2(1.0 + m["jsd"] / self.tau)), self.cap)
        F = 1.0 if m["argmax_flip"] else 0.0
        obj = self.lam[0] * A + self.lam[1] * F - self.lam[2] * ood_penalty(p)
        rec = PairedRecord(p, rt, dt, m, self.n_evals if step is None else step, valid, obj)
        self.cache[k] = rec
        self.n_evals += 1
        hb = HEARTBEAT["fn"]
        if hb is not None:
            hb()
        return rec


@dataclass
class SearchResult:
    strategy: str
    records: list[PairedRecord]
    curve: list[float]                  # best valid JSD so far, per evaluation

    @property
    def best(self) -> float:
        return max((r.jsd for r in self.records if r.valid), default=0.0)

    def top(self, k: int = 5) -> list[PairedRecord]:
        v = sorted((r for r in self.records if r.valid), key=lambda r: -r.jsd)
        return v[:k]

    def best_at(self, budget: int) -> float:
        if not self.curve:
            return 0.0
        return self.curve[min(budget, len(self.curve)) - 1]


def _fresh(oracle, make, tries=30):
    for _ in range(tries):
        p = make()
        if p.key() not in oracle.cache:
            return p
    return make()


def run_search(strategy: str, oracle: PairedOracle, seeds: list[Prompt], budget: int,
               rng: np.random.Generator, mutator: Mutator, population: int = 16,
               explore_frac: float = 0.25, eps: float = 0.2) -> SearchResult:
    start = oracle.n_evals
    records: list[PairedRecord] = []
    curve: list[float] = []
    best = 0.0

    def do(p: Prompt):
        nonlocal best
        if oracle.n_evals - start >= budget and p.key() not in oracle.cache:
            return None
        rec = oracle.evaluate(p)
        records.append(rec)
        if rec.valid:
            best = max(best, rec.jsd)
        curve.append(best)
        return rec

    def full():
        return oracle.n_evals - start >= budget

    if strategy == "benchmark":
        pool = list(seeds)
        for s in seeds:
            pool.extend(make_variants(s, 4))
        for p in pool:
            if full() or do(p) is None:
                break
    elif strategy == "random":
        guard = 0
        while not full() and guard < budget * 20:
            guard += 1
            base = seeds[int(rng.integers(len(seeds)))]
            p = _fresh(oracle, lambda: _mutate_n(base, mutator, int(rng.integers(0, 4))))
            do(p)
    elif strategy == "guided":
        # stage 1: broad exploration (seeds + light mutations), stage 2: exploit best parents
        n_explore = max(1, int(budget * explore_frac))
        guard = 0
        while oracle.n_evals - start < n_explore and guard < budget * 20:
            guard += 1
            base = seeds[int(rng.integers(len(seeds)))]
            p = base if base.key() not in oracle.cache and rng.random() < 0.5 else _fresh(
                oracle, lambda: _mutate_n(base, mutator, 1))
            do(p)
        guard = 0
        while not full() and guard < budget * 20:
            guard += 1
            pop = sorted((r for r in records if r.valid), key=lambda r: -r.objective)[:population]
            if not pop or rng.random() < eps:
                base = seeds[int(rng.integers(len(seeds)))]
                p = _fresh(oracle, lambda: _mutate_n(base, mutator, 1))
            else:
                cand = [pop[int(rng.integers(len(pop)))] for _ in range(3)]       # tournament
                parent = max(cand, key=lambda r: r.objective).prompt
                p = _fresh(oracle, lambda: _mutate_n(parent, mutator, 1 + int(rng.random() < 0.3)))
            do(p)
    else:
        raise ValueError(strategy)
    return SearchResult(strategy, records, curve)


def _mutate_n(p: Prompt, mutator: Mutator, n: int) -> Prompt:
    for _ in range(n):
        p = mutator.mutate(p)
    return p
