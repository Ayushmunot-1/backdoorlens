"""backdoorlens CLI.  Exit codes of `gate`: 0 allow, 10 quarantine, 20 block, 30 inconclusive, 40 re-attest."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from . import __version__


def _pol(a):
    """Policy overrides: inline JSON, or @file.json (the cross-shell-safe form)."""
    from .decision import Policy
    raw = getattr(a, "policy", None)
    if raw and raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    return Policy(**json.loads(raw)) if raw else Policy()


def _kw(a) -> dict:
    """Backend keyword arguments from the command line.

    Three ways to pass them, because quoting JSON on a command line is shell-dependent and was a real
    source of failures (Windows PowerShell strips the double quotes of an inline JSON argument):
      --adapter-base DIR        shorthand for {"adapter_base_path": DIR}  (the common case)
      --backend-kwargs @FILE    read the JSON object from a file
      --backend-kwargs '{...}'  inline JSON (works as before in bash/zsh/cmd)
    """
    raw = getattr(a, "backend_kwargs", None)
    if raw and raw.startswith("@"):
        raw = Path(raw[1:]).read_text(encoding="utf-8")
    kw = json.loads(raw) if raw else {}
    if getattr(a, "adapter_base", None):
        kw["adapter_base_path"] = a.adapter_base
    return kw


def _pin(v):
    """--pin accepts the public key as hex, or the path of the .hex file `keygen --public` wrote
    (so the same command works in bash, PowerShell and cmd without command substitution)."""
    if v and Path(v).is_file():
        return Path(v).read_text(encoding="utf-8").strip()
    return v


def cmd_manifest(a):
    from .scope import build_manifest
    m = build_manifest(a.bundle)
    print(json.dumps(m.to_dict(), indent=1))


def cmd_diff(a):
    from .bundle_diff import diff_bundles
    print(json.dumps(diff_bundles(a.reference, a.derived).to_dict(), indent=1))


def cmd_keygen(a):
    from .attestation import keygen
    print("public key (hex, pin this in your registry):", keygen(a.private, a.public))


def cmd_calibrate(a):
    from .audit import AuditConfig
    from .backends import load_backend
    from .calibration import fit_null, make_spec
    from .probes import build_seed_corpus, suite_id
    from .scope import build_manifest
    kw = _kw(a)
    seeds = build_seed_corpus(a.n_seeds, a.corpus_seed)
    ref = load_backend(a.backend, a.reference, **kw)
    hon = {Path(p).name: load_backend(a.backend, p, **kw) for p in a.honest}
    m = build_manifest(a.honest[0])
    spec = make_spec(a.model_family, m.components["derivation"].detail, m.components["runtime"].hash or "", "own", suite_id(seeds))
    n = fit_null(ref, hon, seeds, a.budget, a.replicates, tuple(a.strategies), a.seed, "own", spec,
                 progress=lambda n_, s_, r_: print(f"  null {n_} {s_} rep {r_}", file=sys.stderr))
    h = n.save(a.out)
    print(json.dumps({"saved": a.out, "hash": h, **n.summary("guided")}, indent=1))


def cmd_attest(a):
    from .audit import AuditConfig, run_audit
    from .calibration import NullDistribution
    kw = _kw(a)
    null = NullDistribution.load(a.null) if a.null else None
    # A null already records the family, template mode and budget it was fitted with. Default to those
    # instead of making the caller restate them: restating is how an audit silently ends up
    # INCONCLUSIVE ("no benign null matches this transformation family") over a typo.
    family, budget = a.model_family, a.budget
    if null is not None:
        if family is None:
            family = null.spec.get("model_family")
            print(f"[attest] using --model-family {family!r} from the supplied null", file=sys.stderr)
        if budget is None:
            budget = int(null.search_budget)
            print(f"[attest] using --budget {budget} from the supplied null", file=sys.stderr)
    family = family if family is not None else "unknown"
    budget = budget if budget is not None else 256
    cfg = AuditConfig(backend=a.backend, budget=budget, n_seeds=a.n_seeds, strategy=a.strategy, seed=a.seed,
                      corpus_seed=a.corpus_seed, model_family=family, backend_kwargs=kw,
                      component_null_path=a.component_null)
    att = run_audit(a.reference, a.derived, cfg, _pol(a), null, out_dir=a.out, signing_key=a.sign)
    d = att["decision"]
    print(json.dumps({"verdict": d["verdict"], "action": d["action"], "reasons": d["reasons"], "flags": d["flags"],
                      "attestation": str(Path(a.out) / "attestation.json")}, indent=1))


def cmd_verify(a):
    from . import attestation as at
    att = at.load(a.attestation)
    ok, msg = at.verify_signature(att, _pin(a.pin))
    print(json.dumps({"signature_ok": ok, "message": msg, "id": att.get("id"), "verdict": att["decision"]["verdict"]}))
    return 0 if ok else 1


def cmd_gate(a):
    from . import attestation as at, monitor
    att = at.load(a.attestation)
    g = monitor.gate(att, a.bundle, _pin(a.pin))
    print(json.dumps(g, indent=1))
    return g["exit_code"]


def cmd_canary(a):
    from . import attestation as at, monitor
    from .backends import load_backend
    from .probes import build_seed_corpus
    att = at.load(a.attestation)
    kw = _kw(a)
    b = load_backend(a.backend, a.bundle, **kw)
    seeds = build_seed_corpus(att["audit_config"]["n_seeds"], att["audit_config"].get("corpus_seed", 0))
    r = monitor.compare_fingerprint(att["canary_fingerprints"], b, seeds[: att["audit_config"]["canary_n"]])
    print(json.dumps(r, indent=1))
    return 0 if r["state"] == "MATCH" else 40


def cmd_replay(a):
    from . import attestation as at, divergence as dv
    from .backends import load_backend
    from .probes import Prompt
    att = at.load(a.attestation)
    kw = _kw(a)
    ref, der = load_backend(a.backend, a.reference, **kw), load_backend(a.backend, a.derived, **kw)
    out = []
    for c in att["counterexamples"]:
        p = Prompt(**{k: c["prompt"][k] for k in ("id", "family", "messages", "lineage")})
        j = dv.jsd(ref.logprobs(ref.render(p.messages)), der.logprobs(der.render(p.messages)))
        out.append({"id": p.id, "stored_jsd": c["jsd"], "replayed_jsd": j, "ok": abs(j - c["jsd"]) <= max(1e-6, 0.01 * c["jsd"])})
    print(json.dumps(out, indent=1))
    return 0 if all(x["ok"] for x in out) else 1


def main(argv=None):
    ap = argparse.ArgumentParser("backdoorlens", description=__doc__)
    ap.add_argument("--version", action="version", version=__version__)
    sp = ap.add_subparsers(dest="cmd", required=True)
    def common(p, ref=False):
        p.add_argument("--backend", default="hf", choices=["hf", "toy"])
        p.add_argument("--backend-kwargs", default=None,
                       help='JSON object, or @file.json to read it from a file, e.g. {"dtype":"float32"}')
        p.add_argument("--adapter-base", default=None, metavar="DIR",
                       help="local base-model directory for an ADAPTER-ONLY bundle (shorthand for "
                            "adapter_base_path; never resolved from adapter_config.json)")
    p = sp.add_parser("manifest"); p.add_argument("bundle"); p.set_defaults(f=cmd_manifest)
    p = sp.add_parser("diff"); p.add_argument("reference"); p.add_argument("derived"); p.set_defaults(f=cmd_diff)
    p = sp.add_parser("keygen"); p.add_argument("--private", required=True); p.add_argument("--public"); p.set_defaults(f=cmd_keygen)
    p = sp.add_parser("calibrate"); common(p)
    p.add_argument("--reference", required=True); p.add_argument("--honest", nargs="+", required=True)
    p.add_argument("--out", required=True); p.add_argument("--model-family", required=True)
    p.add_argument("--budget", type=int, default=256); p.add_argument("--replicates", type=int, default=15)
    p.add_argument("--n-seeds", type=int, default=128); p.add_argument("--corpus-seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0); p.add_argument("--strategies", nargs="+", default=["guided", "random", "benchmark"])
    p.set_defaults(f=cmd_calibrate)
    p = sp.add_parser("attest"); common(p)
    p.add_argument("--reference", required=True); p.add_argument("--derived", required=True); p.add_argument("--out", required=True)
    p.add_argument("--null"); p.add_argument("--sign")
    p.add_argument("--model-family", default=None,
                   help="defaults to the supplied --null's own model_family (it must match, or the "
                        "audit is INCONCLUSIVE); 'unknown' if no null is given")
    p.add_argument("--component-null", default=None,
                   help="intra-weight component null from scripts/component_null_audit.py "
                        "(component_null_audit.json). Records which tensor groups the derived artifact "
                        "modifies that NO honest derivation of the declared family modifies.")
    p.add_argument("--budget", type=int, default=None, help="defaults to the supplied --null's search_budget"); p.add_argument("--n-seeds", type=int, default=128)
    p.add_argument("--strategy", default="guided", choices=["guided", "random", "benchmark"])
    p.add_argument("--seed", type=int, default=0); p.add_argument("--corpus-seed", type=int, default=0)
    p.add_argument("--policy", default=None, help="JSON overrides of Policy fields, or @file.json"); p.set_defaults(f=cmd_attest)
    p = sp.add_parser("verify"); p.add_argument("attestation"); p.add_argument("--pin", help="public key hex, or the path of the .hex file written by keygen --public"); p.set_defaults(f=cmd_verify)
    p = sp.add_parser("gate"); p.add_argument("attestation"); p.add_argument("bundle"); p.add_argument("--pin", help="public key hex, or the path of the .hex file written by keygen --public"); p.set_defaults(f=cmd_gate)
    p = sp.add_parser("canary"); common(p); p.add_argument("attestation"); p.add_argument("bundle"); p.set_defaults(f=cmd_canary)
    p = sp.add_parser("replay"); common(p); p.add_argument("attestation")
    p.add_argument("--reference", required=True); p.add_argument("--derived", required=True); p.set_defaults(f=cmd_replay)
    a = ap.parse_args(argv)
    rc = a.f(a)
    sys.exit(rc or 0)


if __name__ == "__main__":
    main()
