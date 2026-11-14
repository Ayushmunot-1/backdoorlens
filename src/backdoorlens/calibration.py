"""Benign-divergence null, decomposed into LEVEL and SHAPE and fitted for the statistic actually tested.

Why the decomposition: every honest derivation has its own overall drift level (e.g. INT4 with a
different group alignment). A pooled-null test would treat that derivation-to-derivation variance as
evidence of tampering. We therefore test
  * LEVEL  - the artifact's median per-prompt JSD, against the spread of levels across honest
             derivations (t prediction interval, df = k-1), and
  * SHAPE  - per-prompt JSD and the search maximum, each divided by the artifact's own level
             (relative heavy tails), against the same quantities of honest derivations.
The search null is produced by running the identical search procedure at the identical budget on
honest derivations; a percentile of per-prompt JSD is not a valid threshold for a searched maximum.
"""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
from . import divergence as dv
from .probes import Prompt, Mutator, suite_id
from .scope import canonical_json
from .search import PairedOracle, run_search

SPEC_FIELDS = ("model_family", "tool", "method", "bits", "group_size", "calibration_dataset",
               "target_format", "runtime_hash", "template_mode", "suite_id")
EPS = 1e-12


def make_spec(model_family: str, derivation: dict, runtime_hash: str, template_mode: str, suite: str) -> dict:
    d = derivation or {}
    return {"model_family": model_family, "tool": d.get("tool"), "method": d.get("method"),
            "bits": d.get("bits"), "group_size": d.get("group_size"),
            "calibration_dataset": d.get("calibration_dataset"), "target_format": d.get("target_format"),
            "runtime_hash": runtime_hash, "template_mode": template_mode, "suite_id": suite}


def spec_key(spec: dict) -> str:
    return hashlib.sha256(canonical_json({k: spec.get(k) for k in SPEC_FIELDS})).hexdigest()[:20]


def spec_mismatch(null_spec: dict, required: dict) -> dict:
    """Which SPEC_FIELDS actually differ between a supplied null and the audit's required spec.

    Without this, a mismatch surfaces only as "no benign null matches this transformation family",
    which does not say WHICH field is wrong -- e.g. an audit run with --model-family smollm2-135m
    against a null fitted as smollm2-135m-instruct looks identical to a genuinely wrong-family null.
    """
    out = {}
    for k in SPEC_FIELDS:
        a, b = (null_spec or {}).get(k), (required or {}).get(k)
        if a != b:
            out[k] = {"null": a, "required": b}
    return out


def level_of(scores) -> float:
    return max(float(np.median(np.asarray(scores, float))), EPS)


@dataclass
class NullDistribution:
    spec: dict
    key: str
    search_budget: int
    numeric_floor: float
    scores_by_deriv: dict = field(default_factory=dict)          # name -> [per-prompt jsd]
    maxima_by_deriv: dict = field(default_factory=dict)          # name -> {strategy: [best_jsd / level per replicate]}
    created: str = ""
    meta: dict = field(default_factory=dict)

    # --- level (derivation-level) -------------------------------------------------
    def levels(self, exclude: str | None = None) -> np.ndarray:
        return np.array([level_of(v) for k, v in self.scores_by_deriv.items() if k != exclude])

    # --- spike (derivation-level, worst single prompt relative to that derivation's own level) ---
    # A sparse/rare conditional effect (e.g. a trigger that fires on 1-2 of 128 prompts) barely moves
    # the suite-wide median (`level`), because the other ~126 unaffected prompts dominate the median.
    # The single worst prompt is not diluted this way. This uses only data already collected for
    # `levels()` (scores_by_deriv), so it costs no additional model inference to calibrate or evaluate.
    def spikes(self, exclude: str | None = None) -> np.ndarray:
        out = []
        for k, v in self.scores_by_deriv.items():
            if k == exclude:
                continue
            arr = np.asarray(v, float)
            out.append(float(arr.max()) / level_of(arr))
        return np.array(out)

    # --- shape (relative) -----------------------------------------------------------
    def pooled_scores(self, exclude: str | None = None) -> np.ndarray:
        v = [np.asarray(s, float) for k, s in self.scores_by_deriv.items() if k != exclude]
        return np.concatenate(v) if v else np.array([])

    def pooled_norm_scores(self, exclude: str | None = None) -> np.ndarray:
        v = [np.asarray(s, float) / level_of(s) for k, s in self.scores_by_deriv.items() if k != exclude]
        return np.concatenate(v) if v else np.array([])

    def pooled_maxima(self, strategy: str, exclude: str | None = None) -> np.ndarray:
        v = [np.asarray(m.get(strategy, []), float) for k, m in self.maxima_by_deriv.items() if k != exclude]
        return np.concatenate(v) if v else np.array([])

    def maxima_groups(self, strategy: str) -> list:
        return [list(m.get(strategy, [])) for m in self.maxima_by_deriv.values()]

    def max_ratio(self, strategy: str) -> float:
        m = self.pooled_maxima(strategy)
        return float(m.max()) if m.size else float("nan")

    def n_derivations(self, exclude: str | None = None) -> int:
        return len([k for k in self.scores_by_deriv if k != exclude])

    def leave_one_out(self, name: str) -> "NullDistribution":
        return NullDistribution(self.spec, self.key, self.search_budget, self.numeric_floor,
                                {k: v for k, v in self.scores_by_deriv.items() if k != name},
                                {k: v for k, v in self.maxima_by_deriv.items() if k != name},
                                self.created, dict(self.meta, loo_excluded=name))

    def to_dict(self) -> dict:
        return {"spec": self.spec, "key": self.key, "search_budget": self.search_budget,
                "numeric_floor": self.numeric_floor, "scores_by_deriv": self.scores_by_deriv,
                "maxima_by_deriv": self.maxima_by_deriv, "created": self.created, "meta": self.meta}

    @staticmethod
    def from_dict(d: dict) -> "NullDistribution":
        return NullDistribution(d["spec"], d["key"], d["search_budget"], d["numeric_floor"],
                                d["scores_by_deriv"], d["maxima_by_deriv"], d.get("created", ""), d.get("meta", {}))

    def save(self, path) -> str:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict()))
        return self.content_hash()

    @staticmethod
    def load(path) -> "NullDistribution":
        return NullDistribution.from_dict(json.loads(Path(path).read_text()))

    def content_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.to_dict())).hexdigest()[:20]

    def summary(self, strategy: str = "guided") -> dict:
        n, m, lv, sp = self.pooled_norm_scores(), self.pooled_maxima(strategy), self.levels(), self.spikes()
        return {"key": self.key, "n_derivations": self.n_derivations(), "n_prompt_scores": int(n.size),
                "n_search_maxima": int(m.size), "search_budget": self.search_budget,
                "numeric_floor": self.numeric_floor,
                "honest_levels": [float(x) for x in lv],
                "honest_spikes": [float(x) for x in sp],
                "spike_ratio_max": float(sp.max()) if sp.size else None,
                "rel_score_q95": float(np.quantile(n, 0.95)) if n.size else None,
                "rel_score_q995": float(np.quantile(n, 0.995)) if n.size else None,
                "search_ratio_q50": float(np.quantile(m, 0.5)) if m.size else None,
                "search_ratio_q99": float(np.quantile(m, 0.99)) if m.size else None,
                "search_ratio_max": float(m.max()) if m.size else None,
                "min_resolvable_p": (1.0 / (m.size + 1)) if m.size else None}


def numeric_floor(ref, prompts: list[Prompt], n: int = 32) -> float:
    """Largest JSD between two executions of the SAME reference on the same inputs."""
    worst = 0.0
    for p in prompts[:n]:
        t = ref.render(p.messages)
        worst = max(worst, dv.jsd(ref.logprobs(t), ref.logprobs(t)))
    return float(worst)


def stage_a_scores(ref, der, prompts: list[Prompt], template_mode: str = "own") -> list[float]:
    o = PairedOracle(ref, der, template_mode)
    return [o.evaluate(p).jsd for p in prompts]


def fit_null(ref, honest: dict, seeds: list[Prompt], budget: int, replicates: int,
             strategies=("guided", "random", "benchmark"), seed: int = 0, template_mode: str = "own",
             spec: dict | None = None, progress=None, ckpt_dir=None, log=None) -> NullDistribution:
    """honest: {name: Backend} honest derivations of ONE declared transformation family.
    ckpt_dir: if given, each finished derivation is checkpointed so an interrupted run resumes without
    recomputation (results are identical: every random stream depends only on seed|name|strategy|replicate)."""
    scores: dict = {}
    maxima: dict = {}
    tag = hashlib.sha256(json.dumps([budget, replicates, list(strategies), seed, template_mode, suite_id(seeds),
                                     (spec or {}).get("model_family")]).encode()).hexdigest()[:12]
    ck = Path(ckpt_dir) if ckpt_dir else None
    if ck:
        ck.mkdir(parents=True, exist_ok=True)
    for name, d in honest.items():
        f = (ck / f"{name}.{tag}.json") if ck else None
        if f is not None and f.exists():
            data = json.loads(f.read_text())
            scores[name], maxima[name] = data["scores"], data["maxima"]
            if log:
                log(f"      null[{name}] restored from checkpoint")
            continue
        if log:
            log(f"      null[{name}] stage A: {len(seeds)} paired evaluations ...")
        scores[name] = stage_a_scores(ref, d, seeds, template_mode)
        lvl = level_of(scores[name])
        maxima[name] = {}
        for strat in strategies:
            best = []
            reps = 1 if strat == "benchmark" else replicates       # benchmark order is deterministic
            for r in range(reps):
                h = int(hashlib.sha256(f"{seed}|{name}|{strat}|{r}".encode()).hexdigest()[:8], 16)
                rng = np.random.default_rng(h)
                oracle = PairedOracle(ref, d, template_mode, tau=lvl)
                res = run_search(strat, oracle, seeds, budget, rng, Mutator(rng, getattr(ref, "vocab_sample", None)))
                best.append(res.best / lvl)
                if progress:
                    progress(name, strat, r)
                if log:
                    log(f"      null[{name}] {strat} replicate {r + 1}/{reps} done")
            maxima[name][strat] = best
        if f is not None:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps({"scores": scores[name], "maxima": maxima[name]}))
            tmp.replace(f)
        if log:
            log(f"      null[{name}] done ({len(scores)}/{len(honest)} derivations)")
    spec = spec or {}
    return NullDistribution(spec, spec_key(spec) if spec else "unspecified", budget,
                            numeric_floor(ref, seeds), scores, maxima,
                            dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                            {"seed": seed, "replicates": replicates, "strategies": list(strategies),
                             "suite_id": suite_id(seeds)})


def min_maxima_for_alpha(alpha: float) -> int:
    return int(math.ceil(1.0 / alpha)) - 1
