"""End-to-end audit orchestration: manifest -> bundle diff -> paired runs -> search -> confirm -> decide -> attest."""
from __future__ import annotations
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path
import numpy as np
from . import divergence as dv, stats as st
from .attestation import build as build_att, sign as sign_att, save as save_att, check_signing_key
from .backends import load_backend
from .bundle_diff import diff_bundles
from .calibration import NullDistribution, make_spec, spec_key, spec_mismatch, level_of
from .decision import Policy, decide
from .monitor import fingerprint
from .probes import build_seed_corpus, suite_id, make_variants, Mutator, Prompt
from .scope import build_manifest, comparability
from .search import PairedOracle, run_search


@dataclass
class AuditConfig:
    backend: str = "toy"
    budget: int = 512
    n_seeds: int = 128
    strategy: str = "guided"
    seed: int = 0                  # search / sampling seed
    corpus_seed: int = 0           # probe-suite selection seed: MUST match the seed used to fit the null
    template_mode: str = "own"
    top_k_confirm: int = 5
    n_variants: int = 4
    canary_n: int = 50
    gen_n: int = 24                 # prompts for output-level generation comparison
    max_new_tokens: int = 24
    model_family: str = "unknown"
    backend_kwargs: dict = field(default_factory=dict)
    component_null_path: str | None = None   # optional intra-weight component null (see weight_delta.py,
                                             # scripts/component_null_audit.py). When supplied, run_audit
                                             # records which tensor groups the derived artifact modifies
                                             # that NO honest derivation of the declared family modifies.
                                             # Left None, nothing is computed and no verdict changes.

    def to_dict(self):
        return asdict(self)


def stage_a(oracle: PairedOracle, prompts: list[Prompt], null: NullDistribution | None) -> tuple[dict, list]:
    recs = [oracle.evaluate(p) for p in prompts]
    j = np.array([r.jsd for r in recs])
    level = level_of(j)
    spike_ratio = float(j.max()) / level
    out = {"n": len(recs), "level_median_jsd": level, "mean_jsd": float(j.mean()), "max_jsd": float(j.max()),
           "spike_ratio": spike_ratio,
           "argmax_flip_rate": float(np.mean([r.metrics["argmax_flip"] for r in recs])),
           "p_level": None, "level_ratio_vs_max_honest": None, "p_binom": None, "p_minbonf": None,
           "frac_exceed": None, "k_exceed": None, "p_spike": None, "spike_ratio_vs_max_honest": None}
    if null is not None and null.n_derivations() >= 1:
        lv = null.levels()
        out["p_level"] = st.level_prediction_pvalue(lv, level)
        out["level_ratio_vs_max_honest"] = float(level / max(lv.max(), 1e-12))
        # Spike test: the single worst prompt relative to this artifact's own level, calibrated the
        # SAME way (t prediction interval, df = k-1) as the level test, against honest derivations'
        # own max/level ratios. Targets a sparse/rare conditional effect that the suite-wide median
        # ("level") dilutes across ~128 mostly-unaffected prompts.
        sp = null.spikes()
        out["p_spike"] = st.level_prediction_pvalue(sp, spike_ratio)
        out["spike_ratio_vs_max_honest"] = float(spike_ratio / max(sp.max(), 1e-12)) if sp.size else None
        nn = null.pooled_norm_scores()
        if nn.size:
            p = st.conformal_pvalues(nn, j / level)
            k = int((p <= 0.05).sum())
            out.update(k_exceed=k, frac_exceed=k / len(j), p_binom=st.binom_exceed_pvalue(k, len(j), 0.95),
                       p_minbonf=st.bonferroni_min_p(p))
    return out, recs


def output_level(ref, der, prompts, n, max_new, mode) -> dict:
    """Black-box-style comparison on generated text (the 'output-only' baseline evidence)."""
    flips, eds = [], []
    for p in prompts[:n]:
        rt = ref.render(p.messages)
        dtxt = der.render(p.messages) if mode == "own" else rt
        a, b = ref.generate(rt, max_new), der.generate(dtxt, max_new)
        flips.append(dv.structural_flip(a, b))
        eds.append(dv.edit_distance_ratio(a, b))
    return {"n": len(flips), "structural_flip_rate": float(np.mean(flips)) if flips else 0.0,
            "mean_edit_distance": float(np.mean(eds)) if eds else 0.0,
            "exact_match_rate": float(np.mean([e == 0.0 for e in eds])) if eds else 1.0}


def confirm(oracle_ref, ref, der, top, null, policy, mode, n_variants, max_new, tol, level, strategy="guided"):
    """Persistence across rule-based semantic variants + deterministic replay of the top candidates."""
    results = []
    scores = null.pooled_norm_scores() if null is not None else np.array([])
    maxima = null.pooled_maxima(strategy) if null is not None else np.array([])
    q95 = float(np.quantile(scores, 0.95)) if scores.size else float("inf")
    for rec in top:
        variants = make_variants(rec.prompt, n_variants)
        vj = [oracle_ref.evaluate(v).jsd for v in variants]
        if scores.size:
            vp = st.conformal_pvalues(scores, np.asarray(vj) / level)
            npass = int((vp <= policy.alpha_variant).sum())
        else:
            npass = 0
        rt = ref.render(rec.prompt.messages)
        dtxt = der.render(rec.prompt.messages) if mode == "own" else rt
        replay = dv.jsd(ref.logprobs(rt), der.logprobs(dtxt))                 # independent re-execution
        replay_ok = abs(replay - rec.jsd) <= max(tol, 0.01 * rec.jsd)
        go, gd = ref.generate(rt, max_new), der.generate(dtxt, max_new)
        results.append({
            "key": rec.prompt.key(), "prompt": rec.prompt.to_dict(), "jsd": rec.jsd, "metrics": rec.metrics,
            "ratio": rec.jsd / level, "p_search": (st.search_prediction_pvalue(null.maxima_groups(strategy), rec.jsd / level) if null is not None else None),
            "variant_jsd": vj, "n_variants": len(variants), "n_variants_pass": npass, "q95_null_relative": q95,
            "replay_jsd": replay, "replay_ok": bool(replay_ok),
            "persistent": bool(npass >= policy.persist_min_variants and replay_ok),
            "reference_output": go, "derived_output": gd,
            "structural_flip": dv.structural_flip(go, gd), "edit_distance": dv.edit_distance_ratio(go, gd),
            "rendered_reference": rec.ref_text[:600], "rendered_derived": rec.der_text[:600]})
    return results


def run_audit(ref_dir, der_dir, cfg: AuditConfig, policy: Policy, null: NullDistribution | None,
              out_dir=None, signing_key: str | None = None) -> dict:
    # Validate the signing key BEFORE any model work: signing happens after the audit but before the
    # attestation is written, so a bad key would otherwise throw away a completed multi-minute run.
    if signing_key:
        check_signing_key(signing_key)
    ref_m, der_m = build_manifest(ref_dir), build_manifest(der_dir)
    bd = diff_bundles(ref_dir, der_dir, der_m.declared_changes)

    # DOMAIN CHECK, before anything expensive. If the two bundles are not a reference/derived pair of
    # one model (different vocabulary size, different model_type), no paired divergence between them
    # is defined: the right answer is to refuse the measurement and say so, not to report a large
    # number as if behaviour had been compared. Decided from the manifests in ~1s with no model load,
    # and the verdict is left to decision.decide() -- this records a scope gap, it does not choose an
    # outcome. An artifact that is out of scope is not the same finding as one that is divergent.
    cmp_arch = comparability(ref_m, der_m)
    if cmp_arch["status"] == "incomparable":
        detail = "; ".join(f"{k}: reference={v['reference']!r} derived={v['derived']!r}"
                           for k, v in cmp_arch["fields"].items())
        ev = {
            "completeness": der_m.completeness,
            "scope_gaps": [f"reference and derived bundles are not comparable ({detail}): no paired "
                           f"behavioural differential is defined, so none was measured"],
            "adapter_scope": None, "comparability": cmp_arch,
            "bundle": {"n_behavior_diffs": len(bd.behavior_diffs),
                       "n_undeclared_behavior_diffs": len(bd.undeclared_behavior_diffs),
                       "n_rendered_mismatches": len(bd.rendered_mismatches),
                       "n_render_errors": len(bd.render_errors)},
            "null": None,
            "null_note": "no null was selected: the artifacts are not comparable",
            "numeric_floor": 0.0,
            "behavioural_measurement": "not_performed",
            "stage_a": None, "output_level": None, "confirmation": [],
            "search": {"strategy": cfg.strategy, "budget": cfg.budget, "budget_used": 0,
                       "best_jsd": None, "best_ratio": None, "p_search": None,
                       "p_search_pooled_descriptive": None, "level": None, "curve": []},
        }
        dec = decide(ev, policy)
        att = build_att(ref_m.to_dict(), der_m.to_dict(), ev, dec.to_dict(), policy.to_dict(),
                        cfg.to_dict(), None, None, [], {}, {}, bd.to_dict())
        if signing_key:
            att = sign_att(att, signing_key)
        if out_dir:
            od = Path(out_dir)
            od.mkdir(parents=True, exist_ok=True)
            save_att(att, od / "attestation.json")
        return att

    ref = load_backend(cfg.backend, str(ref_dir), **cfg.backend_kwargs)
    der = load_backend(cfg.backend, str(der_dir), **cfg.backend_kwargs)

    # Adapter correctness: if adapter files are present in a bundle but the loaded backend
    # did not actually apply them, the evaluation below reflects a DIFFERENT artifact than
    # the one declared in the scope tuple. This must never be silently ignored, so it is
    # recorded and surfaced as a scope gap that decision.py treats as blocking ATTESTED.
    scope_gaps = []
    adapter_scope = {"reference": {"status": getattr(ref, "adapter_status", "none"),
                                    "detail": getattr(ref, "adapter_detail", {})},
                      "derived": {"status": getattr(der, "adapter_status", "none"),
                                  "detail": getattr(der, "adapter_detail", {})}}
    for who, b in (("reference", ref), ("derived", der)):
        if getattr(b, "adapter_status", "none") == "unsupported":
            reason = getattr(b, "adapter_detail", {}).get("reason", "adapter present but not applied")
            scope_gaps.append(f"{who} bundle: adapter present but not applied by the backend ({reason})")
    if scope_gaps and der_m.completeness == "complete":
        der_m.completeness = "partial"

    seeds = build_seed_corpus(cfg.n_seeds, cfg.corpus_seed)
    sid = suite_id(seeds)

    spec = make_spec(cfg.model_family, der_m.components["derivation"].detail,
                     der_m.components["runtime"].hash or "", cfg.template_mode, sid)
    null_used, null_note = null, None
    if null is not None and null.key != spec_key(spec):
        diff = spec_mismatch(null.spec, spec)
        detail = "; ".join(f"{k}: null={v['null']!r} required={v['required']!r}" for k, v in diff.items())
        null_used, null_note = None, (
            f"supplied null {null.key} does not match required family key {spec_key(spec)}"
            + (f" -- differing field(s): {detail}" if detail else ""))

    base_cost_r, base_cost_d = ref.cost.snapshot(), der.cost.snapshot()

    sa_oracle = PairedOracle(ref, der, cfg.template_mode, 1.0)
    sa, sa_recs = stage_a(sa_oracle, seeds, null_used)
    level = sa["level_median_jsd"]
    out_lvl = output_level(ref, der, seeds, cfg.gen_n, cfg.max_new_tokens, cfg.template_mode)

    rng = np.random.default_rng(int(__import__("hashlib").sha256(f"{cfg.seed}|audit|{cfg.strategy}".encode()).hexdigest()[:8], 16))
    s_oracle = PairedOracle(ref, der, cfg.template_mode, level)
    res = run_search(cfg.strategy, s_oracle, seeds, cfg.budget, rng, Mutator(rng, getattr(ref, "vocab_sample", None)))
    maxima = null_used.pooled_maxima(cfg.strategy) if null_used is not None else np.array([])
    p_search = st.search_prediction_pvalue(null_used.maxima_groups(cfg.strategy), res.best / level) if null_used is not None else None
    p_search_pooled = st.empirical_pvalue_max(maxima, res.best / level) if maxima.size else None

    conf_oracle = PairedOracle(ref, der, cfg.template_mode, level)
    tol = max(1e-6, (null_used.numeric_floor if null_used is not None else 0.0) * 10)
    conf = confirm(conf_oracle, ref, der, res.top(cfg.top_k_confirm), null_used, policy, cfg.template_mode,
                   cfg.n_variants, cfg.max_new_tokens, tol, level, cfg.strategy)

    numeric_floor_v = float(null_used.numeric_floor) if null_used is not None else 0.0

    # Optional intra-weight component differential, calibrated against honest derivations of the
    # declared family (weight_delta.py). Purely additive: with no component null supplied, the key is
    # absent from the evidence and decision.decide() behaves exactly as before.
    comp_anom = None
    if cfg.component_null_path:
        try:
            from . import weight_delta as _wd
            _cnull = json.loads(Path(cfg.component_null_path).read_text())
            # accept either a bare null dict (weight_delta.fit_component_null output) or the full
            # scripts/component_null_audit.py payload, which nests it under "component_null"
            if isinstance(_cnull, dict) and "status" not in _cnull and isinstance(
                    _cnull.get("component_null"), dict):
                _cnull = _cnull["component_null"]
            comp_anom = _wd.component_anomalies(ref_dir, der_dir, _cnull)
        except Exception as e:                     # a missing/unreadable null must not kill an audit
            comp_anom = {"status": "error", "error": f"{type(e).__name__}: {e}"[:200]}

    ev = {
        "completeness": der_m.completeness,
        **({"component_anomaly": comp_anom} if comp_anom is not None else {}),
        "scope_gaps": scope_gaps, "adapter_scope": adapter_scope,
        "bundle": {"n_behavior_diffs": len(bd.behavior_diffs), "n_undeclared_behavior_diffs": len(bd.undeclared_behavior_diffs),
                   "n_rendered_mismatches": len(bd.rendered_mismatches), "n_render_errors": len(bd.render_errors)},
        "null": ({**null_used.summary(cfg.strategy), "n_prompt_scores": int(null_used.pooled_scores().size),
                  "n_search_maxima": int(maxima.size), "search_budget": null_used.search_budget} if null_used is not None else None),
        "null_note": null_note, "numeric_floor": numeric_floor_v,
        "stage_a": sa, "output_level": out_lvl,
        "search": {"strategy": cfg.strategy, "budget": cfg.budget, "budget_used": s_oracle.n_evals,
                   "best_jsd": res.best, "best_ratio": res.best / level, "p_search": p_search, "p_search_pooled_descriptive": p_search_pooled, "level": level, "curve": res.curve},
        "confirmation": conf,
    }
    dec = decide(ev, policy)
    cost = {"reference": {k: ref.cost.snapshot()[k] - base_cost_r[k] for k in base_cost_r},
            "derived": {k: der.cost.snapshot()[k] - base_cost_d[k] for k in base_cost_d},
            "paired_cases_search": s_oracle.n_evals, "paired_cases_stage_a": sa_oracle.n_evals,
            "paired_cases_confirmation": conf_oracle.n_evals}
    canary = fingerprint(der, seeds[: cfg.canary_n])
    cex = [c for c in conf if c["persistent"] or c["jsd"] == max(x["jsd"] for x in conf)] if conf else []
    att = build_att(ref_m.to_dict(), der_m.to_dict(), ev, dec.to_dict(), policy.to_dict(), cfg.to_dict(),
                    null_used.summary(cfg.strategy) if null_used is not None else None,
                    null_used.content_hash() if null_used is not None else None, cex, cost, canary, bd.to_dict())
    if signing_key:
        att = sign_att(att, signing_key)
    if out_dir:
        od = Path(out_dir)
        od.mkdir(parents=True, exist_ok=True)
        save_att(att, od / "attestation.json")
        with open(od / "traces.jsonl", "w") as f:
            for tag, recs in (("stage_a", sa_recs), ("search", res.records)):
                for r in recs:
                    f.write(json.dumps({"stage": tag, "step": r.step, "prompt": r.prompt.to_dict(), "valid": r.valid,
                                        "objective": r.objective, "metrics": r.metrics, "rendered_ref": r.ref_text[:300],
                                        "rendered_der": r.der_text[:300]}) + "\n")
    return att
