"""bl_tools.py -- the build-your-own-artifacts toolbox. One cross-platform CLI (Windows / macOS / Linux).

Why this exists: building honest derivations and tampered test artifacts used to need multi-line
`python -c "..."` blocks. Those are quoted differently by PowerShell, cmd and bash, and they broke for
anyone who was not on the author's shell. Every step is now a plain one-line command:

    python scripts/bl_tools.py doctor
    python scripts/bl_tools.py register       models/SmolLM2-135M-Instruct
    python scripts/bl_tools.py honest-rtn     --ref models/SmolLM2-135M-Instruct --work work/artifacts --bits 4 --count 6 --prefix cal4
    python scripts/bl_tools.py honest-lora    --ref models/SmolLM-135M           --work work/honest_lora --count 6
    python scripts/bl_tools.py component-null --ref models/SmolLM2-135M-Instruct --work work/artifacts --prefix cal4 --out work/component_null_int4
    python scripts/bl_tools.py template       --ref models/SmolLM2-135M-Instruct --out work/artifacts/t_template
    python scripts/bl_tools.py generation     --ref models/SmolLM2-135M-Instruct --out work/artifacts/t_gen
    python scripts/bl_tools.py canary         --ref models/SmolLM2-135M-Instruct --out work/artifacts/t_canary_2.0 --scale 2.0
    python scripts/bl_tools.py trigger        --ref models/SmolLM2-135M-Instruct --out work/artifacts/t_trigger_easy --trigger lantern
    python scripts/bl_tools.py adapter-info   models/third_party_lora
    python scripts/bl_tools.py diff-triage    --ref models/SmolLM-135M --artifact models/third_party_lora --honest-work work/honest_lora
    python scripts/bl_tools.py declare        --src models/third_party_lora --out work/third_party_declared --base-model-dir models/SmolLM-135M
    python scripts/bl_tools.py policy         --out work/policy_adapter.json --set undeclared_diff_is_suspicious=false
    python scripts/bl_tools.py report         work/audit/t_trigger_easy/attestation.json --out work/audit/t_trigger_easy/report.txt
    python scripts/bl_tools.py tamper         --src work/artifacts/t_trigger_easy --out work/artifacts/tampered_copy

Each subcommand is a thin wrapper over a function in src/backdoorlens/derive.py or weight_delta.py; the
arithmetic is theirs, not this file's. Planted behaviours are harmless canned markers.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _need_dir(p, what):
    if not Path(p).is_dir():
        raise SystemExit(f"{what} not found: {Path(p).resolve()}\n"
                         f"(paths are relative to your current directory: {Path.cwd()})")
    return Path(p)


def _refuse_nonexistent_reference(ref):
    ref = _need_dir(ref, "reference model directory")
    if not (ref / "config.json").exists():
        raise SystemExit(f"{ref} has no config.json -- it is not a Hugging Face model directory. "
                         "Download it first (see docs/RUNBOOK.md, step A2).")
    return ref


# ----------------------------------------------------------------------------------------------------
def cmd_doctor(a):
    import importlib
    import platform
    ok = True
    print(f"python {platform.python_version()} on {platform.system()}  (need >= 3.10)")
    if sys.version_info < (3, 10):
        ok = False
        print("  FAIL: python is too old")
    need = [("numpy", True), ("scipy", True), ("jinja2", True), ("cryptography", True),
            ("torch", False), ("transformers", False), ("safetensors", False), ("tokenizers", False),
            ("peft", False), ("matplotlib", False), ("pandas", False), ("pptx", False)]
    print("\ncore packages are required; the rest are needed for real (Hugging Face) models / reports / the deck:")
    for name, core in need:
        try:
            m = importlib.import_module(name)
            print(f"  ok       {name:13s} {getattr(m, '__version__', '')}")
        except Exception as e:                                   # noqa: BLE001
            tag = "MISSING " if core else "missing "
            print(f"  {tag} {name:13s} ({type(e).__name__})")
            ok = ok and not core
    try:
        import torch
        print(f"\ntorch: cuda available = {torch.cuda.is_available()}  "
              f"({'GPU will be used' if torch.cuda.is_available() else 'CPU only: fine for a 135M model'})")
    except Exception:                                            # noqa: BLE001
        pass
    for p in a.paths or []:
        print(f"  {'ok       ' if Path(p).is_dir() else 'MISSING '} {p}")
        ok = ok and Path(p).is_dir()
    print("\nRESULT:", "environment looks fine" if ok else "fix the lines marked MISSING/FAIL, then re-run doctor")
    return 0 if ok else 1


def cmd_register(a):
    from backdoorlens.derive import register_reference
    ref = _refuse_nonexistent_reference(a.ref)
    register_reference(ref)
    print(f"registered {ref}: runtime.json and derivation.json exist (written only if they were absent)")


def cmd_honest_rtn(a):
    from backdoorlens.derive import derive_rtn
    ref = _refuse_nonexistent_reference(a.ref)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    for i in range(a.count):
        out = work / f"{a.prefix}_{i}"
        derive_rtn(ref, out, bits=a.bits, group=a.group, offset=a.offset_start + i)
        print(f"built {out}  (INT{a.bits} round-to-nearest simulation, group {a.group}, offset {a.offset_start + i})")
    print(f"\n{a.count} honest INT{a.bits} derivations in {work}. Next: calibrate (RUNBOOK step A5).")


def cmd_honest_lora(a):
    from backdoorlens.derive import derive_benign_lora_sft
    ref = _refuse_nonexistent_reference(a.ref)
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    for i in range(a.count):
        out = work / f"{a.prefix}_{i}"
        kw = {"target_modules": tuple(t.strip() for t in a.targets.split(",") if t.strip())} if a.targets else {}
        derive_benign_lora_sft(ref, out, rank=a.rank, lora_alpha=a.alpha if a.alpha else a.rank, steps=a.steps, lr=a.lr,
                               seed=a.seed_start + i, **kw)
        print(f"built {out}  (benign LoRA-SFT adapter, rank {a.rank}, seed {a.seed_start + i})")
    print(f"\n{a.count} honest LoRA adapters in {work}. Next: calibrate with --adapter-base {ref}.")


def cmd_template(a):
    """INT4 honest derivation, then a hidden instruction added to the chat template (exactly what the experiment builds)."""
    from backdoorlens.derive import derive_rtn, inject_canary_note
    ref = _refuse_nonexistent_reference(a.ref)
    out = derive_rtn(ref, a.out, bits=a.bits, group=a.group, offset=a.offset)
    tcp = out / "tokenizer_config.json"
    tc = json.loads(tcp.read_text(encoding="utf-8"))
    jp = out / "chat_template.jinja"
    src = jp.read_text(encoding="utf-8") if jp.exists() else (tc.get("chat_template") or "")
    if not src:
        raise SystemExit("the reference has no chat template, so the template-injection tamper cannot be built")
    new = inject_canary_note(src)                      # raises if the edit would be a no-op
    if isinstance(tc.get("chat_template"), str):
        tc["chat_template"] = inject_canary_note(tc["chat_template"])
        tcp.write_text(json.dumps(tc, indent=1), encoding="utf-8")
    if jp.exists():
        jp.write_text(new, encoding="utf-8")
    print(f"built {a.out}  (INT{a.bits} derivation + a hidden instruction in the chat template; the weights are an honest quantization)")


def cmd_generation(a):
    """INT4 honest derivation, then a changed generation_config (default temperature 0.9 + sampling, as in the experiment)."""
    from backdoorlens.derive import derive_rtn
    ref = _refuse_nonexistent_reference(a.ref)
    gen = {"temperature": 0.9, "do_sample": True}
    for kv in a.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        try:
            gen[k] = json.loads(v)
        except json.JSONDecodeError:
            gen[k] = v
    out = derive_rtn(ref, a.out, bits=a.bits, group=a.group, offset=a.offset)
    gp = out / "generation_config.json"
    g = json.loads(gp.read_text(encoding="utf-8")) if gp.exists() else {}
    g.update(gen)
    gp.write_text(json.dumps(g, indent=1), encoding="utf-8")
    print(f"built {a.out}  (INT{a.bits} derivation + generation_config {gen}; weights are an honest quantization)")


def cmd_canary(a):
    from backdoorlens.derive import derive_canary_weight_delta
    ref = _refuse_nonexistent_reference(a.ref)
    derive_canary_weight_delta(ref, a.out, scale=a.scale, fraction=a.fraction, seed=a.seed, bits=a.bits, group=a.group)
    print(f"built {a.out}  (Gaussian noise, relative std {a.scale}, on {100 * a.fraction:.0f}% of weights, then INT{a.bits}). "
          "This is a measuring stick for sensitivity, not an attack.")


def cmd_trigger(a):
    from backdoorlens.derive import plant_trigger_behavior
    ref = _refuse_nonexistent_reference(a.ref)
    plant_trigger_behavior(ref, a.out, trigger=a.trigger, marker=a.marker, steps=a.steps, lr=a.lr,
                           base_bits=a.bits, group=a.group, offset=a.offset, max_rounds=a.max_rounds,
                           batch_size=a.batch_size, require=not a.no_require, n_check=a.n_check)
    truth_path = Path(a.out).parent / "_truth" / (Path(a.out).name + ".json")
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    v = truth.get("validation", {})
    print(f"built {a.out}")
    print(f"  planted behaviour verified : {truth.get('verified')}   "
          f"(trigger->marker {100 * v.get('success_rate', 0):.0f}%, clean->marker {100 * v.get('false_trigger_rate', 0):.0f}%)")
    print(f"  ground truth (kept OUTSIDE the audited bundle): {truth_path}")
    if not truth.get("verified"):
        print("  WARNING: the behaviour did not install, so this artifact is NOT a valid backdoor test case. "
              "Do not count it in any detection result. Raise --steps / --max-rounds / --lr.")


def cmd_adapter(a):
    from backdoorlens.derive import derive_adapter_only
    ref = _refuse_nonexistent_reference(a.ref)
    derive_adapter_only(ref, a.out, rank=a.rank, scale=a.scale, seed=a.seed)
    print(f"built {a.out}  (adapter-ONLY bundle, random non-zero LoRA; audit it with --adapter-base {ref})")


def cmd_adapter_info(a):
    """Read-only look at a downloaded adapter: what it says about itself. Nothing here is trusted or fetched."""
    src = _need_dir(a.src, "adapter directory")
    files = sorted(p.name for p in src.iterdir() if p.is_file())
    print(f"files in {src}: {', '.join(files)}")
    cp = src / "adapter_config.json"
    if not cp.exists():
        print("no adapter_config.json: this does not look like a PEFT/LoRA adapter")
        return 1
    c = json.loads(cp.read_text(encoding="utf-8"))
    for k in ("peft_type", "base_model_name_or_path", "r", "lora_alpha", "lora_dropout", "target_modules", "task_type"):
        print(f"  {k:26s}: {c.get(k)}")
    print("\nbase_model_name_or_path is what the AUTHOR claims; BackdoorLens never downloads or trusts it. "
          "You choose the reference yourself and pass it with --adapter-base.")
    return 0


def cmd_diff_triage(a):
    """Which undeclared bundle differences of an adapter are just 'adapter-only format' and which are the artifact's own?

    An adapter-only bundle has no config/tokenizer/runtime files, so EVERY adapter differs from a full model in
    the same structural ways. Those differences are shared by all honest adapters. A difference that honest
    adapters do NOT have (a shipped tokenizer, a shipped chat template, a changed special token...) is the
    artifact's own and must be reviewed by a human before anyone waives undeclared differences.
    """
    from backdoorlens.bundle_diff import diff_bundles
    ref = _refuse_nonexistent_reference(a.ref)
    art = _need_dir(a.artifact, "artifact")
    honest_dirs = [Path(h) for h in a.honest]
    if a.honest_work:
        w = _need_dir(a.honest_work, "honest work directory")
        honest_dirs += sorted(d for d in w.iterdir() if d.is_dir() and d.name.startswith(a.honest_prefix + "_"))
    if len(honest_dirs) < 2:
        raise SystemExit("need at least 2 honest adapters (use --honest-work DIR --honest-prefix honest_lora)")

    def keyed(d):
        rep = diff_bundles(ref, d)
        return ({(x.component, x.field): json.dumps(x.derived, sort_keys=True, default=str) for x in rep.undeclared_behavior_diffs}, rep)

    mine, rep = keyed(art)
    hon = [keyed(d)[0] for d in honest_dirs]
    shared = sorted(k for k, v in mine.items() if all(h.get(k) == v for h in hon))
    own = sorted(k for k in mine if k not in shared)
    print(f"{art.name}: {len(mine)} undeclared behaviour-relevant differences against {ref.name}")
    print(f"  identical in all {len(hon)} honest adapters (structural, adapter-only format) : {len(shared)}")
    print(f"  NOT shared by the honest adapters (this artifact's own)                       : {len(own)}")
    for c, f in own:
        print(f"      REVIEW  {c}.{f}")
    if rep.rendered_mismatches:
        print(f"  rendered chat-template mismatches: {len(rep.rendered_mismatches)}  -> REVIEW the template by hand")
    if own or rep.rendered_mismatches:
        print("\nRESULT: REVIEW REQUIRED. Do not waive undeclared differences until you have read every line marked REVIEW.")
        return 3
    print("\nRESULT: every undeclared difference is shared with honest adapters, i.e. it comes from the adapter-only "
          "format and not from this artifact. Waiving them (policy file) loses nothing the honest adapters do not also lose.")
    return 0


def cmd_declare(a):
    """Copy a third-party adapter and write the scope files it lacks. Never edits the original."""
    from backdoorlens.derive import write_derivation, runtime_manifest
    src, out = _need_dir(a.src, "adapter directory"), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --out (this command never overwrites)")
    shutil.copytree(src, out, ignore=shutil.ignore_patterns(".cache", ".git", "*.md", ".gitattributes"))
    acfg = {}
    if (out / "adapter_config.json").exists():
        acfg = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    spec = dict(tool=a.tool, method=a.method, declared_changes=["adapters"])
    for k in ("r", "lora_alpha", "lora_dropout", "target_modules"):
        if k in acfg:
            spec["rank" if k == "r" else k] = acfg[k]
    if a.base_model_dir:
        spec["declared_base"] = str(a.base_model_dir)
    write_derivation(out, **spec)
    runtime_manifest(out)
    print(f"wrote {out}\n  derivation.json : {json.dumps(spec)}\n  runtime.json    : default (transformers, float32)")
    print("\nTHIS IS A CLAIM, NOT A MEASUREMENT: you are declaring that this artifact is a "
          f"'{a.method}' derivation. The audit then tests whether its behaviour is consistent with that claim.\n"
          "The method label must equal the label of the honest derivations the null was fitted on.")


def cmd_component_null(a):
    """Which tensor groups do honest derivations of this family modify? Runs no inference."""
    from backdoorlens import weight_delta as wd
    ref = _refuse_nonexistent_reference(a.ref)
    work = _need_dir(a.work, "work directory")
    honest = {d.name: d for d in sorted(work.iterdir()) if d.is_dir() and d.name.startswith(a.prefix + "_")}
    if len(honest) < 5:
        raise SystemExit(f"found {len(honest)} honest derivations named '{a.prefix}_*' in {work}; the policy needs at least 5 "
                         "(it must estimate derivation-to-derivation variation). Build them with honest-rtn first.")
    null = wd.fit_component_null(ref, honest, log=print)
    loo = wd.leave_one_out_component_fpr(ref, honest, log=print)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    payload = {"component_null": null, "leave_one_out": loo, "honest_prefix": a.prefix}
    (out / "component_null_audit.json").write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
    print(f"\nhonest derivations used : {len(honest)}")
    print(f"tensor groups no honest derivation ever modifies : {null.get('n_never_changed_keys')}")
    print(f"leave-one-out false positives : {loo['n_false_positives']}/{loo['n_tested']}")
    print(f"wrote {out / 'component_null_audit.json'}")


def cmd_policy(a):
    """Write a policy-override file for `attest --policy @file`. Valid field names are checked against Policy."""
    from dataclasses import fields
    from backdoorlens.decision import Policy
    valid = {f.name for f in fields(Policy)}
    over = {}
    for kv in a.set:
        if "=" not in kv:
            raise SystemExit(f"--set expects key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in valid:
            raise SystemExit(f"{k!r} is not a policy field. Valid: {', '.join(sorted(valid))}")
        over[k] = json.loads(v)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(over, indent=1), encoding="utf-8")
    print(f"wrote {a.out}: {over}\nThe overrides are recorded inside the certificate, so anyone reading it can see them.")


def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}g}"
    return str(x)


def cmd_report(a):
    """Plain-text summary of ONE certificate: what was compared, what was measured, what was decided."""
    att = json.loads(Path(a.attestation).read_text(encoding="utf-8"))
    d, ev = att.get("decision", {}), att.get("evidence", {})
    sa, se, nul = ev.get("stage_a") or {}, ev.get("search") or {}, ev.get("null") or {}
    ca = ev.get("component_anomaly") or {}
    par = att.get("parameters") or {}
    L = []
    add = L.append
    add(f"BACKDOORLENS CERTIFICATE {att.get('id')}")
    add("=" * 72)
    add(f"VERDICT : {d.get('verdict')}   ->   ACTION: {d.get('action')}")
    for r in d.get("reasons", []):
        add(f"  - {r}")
    if d.get("flags"):
        add(f"  flags: {', '.join(d['flags'])}")
    add("")
    add("WHAT WAS COMPARED")
    for side in ("reference", "derived"):
        w = (par.get(side) or {}).get("weights") or {}
        ad = (par.get(side) or {}).get("adapters") or {}
        arch = ((par.get(side) or {}).get("architecture") or {}).get("fields") or {}
        add(f"  {side:9s} weights: {w.get('total_parameters', 0):,} params in {w.get('n_tensors', 0)} tensors"
            f" | adapters: {ad.get('total_parameters', 0):,} params in {ad.get('n_tensors', 0)} tensors"
            f" | {arch.get('model_type', '?')} {arch.get('num_hidden_layers', '?')} layers, hidden {arch.get('hidden_size', '?')}")
    drv = (att.get("derived") or {}).get("derivation") or {}
    add(f"  declared transformation : {drv.get('tool')} / {drv.get('method')}  bits={drv.get('bits')}  declared_changes={(att.get('derived') or {}).get('declared_changes')}")
    add(f"  scope completeness      : {(att.get('derived') or {}).get('completeness')}")
    add(f"  scope hash (reference)  : {(att.get('reference') or {}).get('scope_hash')}")
    add(f"  scope hash (derived)    : {(att.get('derived') or {}).get('scope_hash')}")
    b = att.get("bundle_diff") or {}
    add(f"  bundle differential     : {b.get('n_behavior_diffs')} behaviour-relevant differences, "
        f"{b.get('n_undeclared_behavior_diffs')} undeclared, {len(b.get('rendered_mismatches') or [])} rendered-template mismatches")
    for x in (b.get("differences") or [])[:6]:
        add(f"      [{'declared  ' if x.get('declared') else 'UNDECLARED'}] {x.get('component')}.{x.get('field')}")
    add("")
    add("HOW 'NORMAL' WAS DEFINED (the matched null)")
    add(f"  null id                 : {(att.get('null') or {}).get('hash')}")
    add(f"  honest derivations used : {nul.get('n_derivations')}   (levels {_fmt(min(nul['honest_levels'])) if nul.get('honest_levels') else 'n/a'}"
        f" .. {_fmt(max(nul['honest_levels'])) if nul.get('honest_levels') else 'n/a'})")
    add(f"  null search budget      : {nul.get('search_budget')}   audit budget used: {se.get('budget_used')}")
    add("")
    if ev.get("behavioural_measurement") == "not_performed":
        add("BEHAVIOURAL MEASUREMENT: NOT PERFORMED (the pair was refused before any model was loaded). "
            "The statistics are absent, not zero.")
    else:
        add("WHAT WAS MEASURED (the artifact's own level is the denominator of every ratio)")
        add(f"  level (median JSD over {sa.get('n')} probes)   : {_fmt(sa.get('level_median_jsd'))}   = {_fmt(sa.get('level_ratio_vs_max_honest'))} x the largest honest level   (p_level {_fmt(sa.get('p_level'))})")
        add(f"  spike ratio (worst probe / level)          : {_fmt(sa.get('spike_ratio'))}   = {_fmt(sa.get('spike_ratio_vs_max_honest'))} x the largest honest spike   (p_spike {_fmt(sa.get('p_spike'))})")
        add(f"  argmax flip rate (top token changed)       : {_fmt(sa.get('argmax_flip_rate'))}   [descriptive]")
        add(f"  search best JSD / ratio                    : {_fmt(se.get('best_jsd'))} / {_fmt(se.get('best_ratio'))}   (p_search {_fmt(se.get('p_search'))}, strategy {se.get('strategy')})")
        ol = ev.get("output_level") or {}
        add(f"  generated-output structural flip rate      : {_fmt(ol.get('structural_flip_rate'))}   mean edit distance {_fmt(ol.get('mean_edit_distance'))}")
    add(f"  component channel                          : {ca.get('status', 'not supplied')}"
        + (f", {ca.get('n_hard_anomalies')} tensor groups changed that no honest derivation changes" if ca.get('status') == 'ok' else ""))
    add("")
    cx = att.get("counterexamples") or []
    add(f"COUNTEREXAMPLES: {len(cx)} recorded")
    for c in cx[:3]:
        add(f"  - {c.get('prompt', {}).get('id')}: JSD {_fmt(c.get('jsd'))}, ratio {_fmt(c.get('ratio'))}, "
            f"{c.get('n_variants_pass')}/{c.get('n_variants')} variants exceed the null, replay_ok={c.get('replay_ok')}")
    add("")
    cost = att.get("cost") or {}
    add(f"COST: {cost.get('paired_cases_stage_a')} stage-A + {cost.get('paired_cases_search')} search + {cost.get('paired_cases_confirmation')} confirmation paired cases")
    add(f"CREATED {att.get('created')}   EXPIRES {att.get('expires')}   policy {att.get('policy_version')}   signed={'yes' if att.get('signature') else 'NO'}")
    add("")
    add("LIMITATIONS (stored in the certificate)")
    for x in att.get("limitations", []):
        add(f"  - {x}")
    text = "\n".join(L)
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {a.out}")


def cmd_tamper(a):
    """Flip one byte of a weight file in a COPY: the post-certification modification test."""
    src, out = _need_dir(a.src, "bundle"), Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} already exists; choose a new --out")
    shutil.copytree(src, out)
    files = sorted(list(out.glob("*.safetensors")) + list(out.glob("adapter_model.*")))
    if not files:
        raise SystemExit(f"no weight or adapter file found in {out} to modify")
    f = files[0]
    b = bytearray(f.read_bytes())
    b[len(b) // 2] ^= 0xFF
    f.write_bytes(bytes(b))
    print(f"copied {src} -> {out} and flipped one byte in {f.name}. The gate must now answer INVALIDATED (exit 40).")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)

    p = sp.add_parser("doctor", help="check Python and packages"); p.add_argument("paths", nargs="*"); p.set_defaults(f=cmd_doctor)
    p = sp.add_parser("register", help="write runtime.json/derivation.json into a downloaded reference")
    p.add_argument("ref"); p.set_defaults(f=cmd_register)

    p = sp.add_parser("honest-rtn", help="build k honest simulated-quantization derivations (the null's raw material)")
    p.add_argument("--ref", required=True); p.add_argument("--work", required=True)
    p.add_argument("--bits", type=int, default=4, choices=[2, 3, 4, 5, 6, 8])
    p.add_argument("--count", type=int, default=6); p.add_argument("--group", type=int, default=128)
    p.add_argument("--prefix", default=None, help="folder prefix; default cal<bits>")
    p.add_argument("--offset-start", type=int, default=0)
    p.set_defaults(f=cmd_honest_rtn)

    p = sp.add_parser("honest-lora", help="train k honest benign LoRA adapters (the null's raw material for adapters)")
    p.add_argument("--ref", required=True); p.add_argument("--work", required=True)
    p.add_argument("--count", type=int, default=6); p.add_argument("--prefix", default="honest_lora")
    p.add_argument("--rank", type=int, default=16); p.add_argument("--steps", type=int, default=60)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--seed-start", type=int, default=0)
    p.add_argument("--alpha", type=int, default=None, help="LoRA alpha; default = rank")
    p.add_argument("--targets", default=None, help="comma-separated target modules, e.g. q_proj,v_proj "
                   "(default: all seven projections). Use the audited adapter's own list (adapter-info) to match it.")
    p.set_defaults(f=cmd_honest_lora)

    p = sp.add_parser("template", help="INT4 derivation + hidden instruction in the chat template")
    p.add_argument("--ref", required=True); p.add_argument("--out", required=True)
    p.add_argument("--bits", type=int, default=4); p.add_argument("--group", type=int, default=128)
    p.add_argument("--offset", type=int, default=50); p.set_defaults(f=cmd_template)

    p = sp.add_parser("generation", help="INT4 derivation + changed generation_config")
    p.add_argument("--ref", required=True); p.add_argument("--out", required=True)
    p.add_argument("--bits", type=int, default=4); p.add_argument("--group", type=int, default=128)
    p.add_argument("--offset", type=int, default=50)
    p.add_argument("--set", action="append", default=[], metavar="key=value",
                   help="override/add generation settings; default temperature=0.9 do_sample=true"); p.set_defaults(f=cmd_generation)

    p = sp.add_parser("canary", help="controlled weight perturbation (sensitivity measuring stick)")
    p.add_argument("--ref", required=True); p.add_argument("--out", required=True)
    p.add_argument("--scale", type=float, required=True); p.add_argument("--fraction", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=50, help="default 50 = the experiment's tamper offset"); p.add_argument("--bits", type=int, default=4)
    p.add_argument("--group", type=int, default=128); p.set_defaults(f=cmd_canary)

    p = sp.add_parser("trigger", help="plant a HARMLESS conditional marker behaviour (ground truth you control)")
    p.add_argument("--ref", required=True); p.add_argument("--out", required=True)
    p.add_argument("--trigger", default="lantern"); p.add_argument("--marker", default="ZXQ-CANARY")
    p.add_argument("--steps", type=int, default=40); p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--bits", type=int, default=4); p.add_argument("--group", type=int, default=128)
    p.add_argument("--offset", type=int, default=50); p.add_argument("--max-rounds", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=8); p.add_argument("--n-check", type=int, default=16)
    p.add_argument("--no-require", action="store_true", help="do not stop if the behaviour failed to install (smoke tests only)")
    p.set_defaults(f=cmd_trigger)

    p = sp.add_parser("adapter", help="adapter-ONLY bundle with a random non-zero LoRA")
    p.add_argument("--ref", required=True); p.add_argument("--out", required=True)
    p.add_argument("--rank", type=int, default=4); p.add_argument("--scale", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0); p.set_defaults(f=cmd_adapter)

    p = sp.add_parser("adapter-info", help="read-only summary of a downloaded adapter's own config")
    p.add_argument("src"); p.set_defaults(f=cmd_adapter_info)

    p = sp.add_parser("diff-triage", help="split an adapter's undeclared differences into structural vs its own")
    p.add_argument("--ref", required=True); p.add_argument("--artifact", required=True)
    p.add_argument("--honest", nargs="*", default=[]); p.add_argument("--honest-work", default=None)
    p.add_argument("--honest-prefix", default="honest_lora"); p.set_defaults(f=cmd_diff_triage)

    p = sp.add_parser("declare", help="copy a third-party adapter and declare its transformation scope")
    p.add_argument("--src", required=True); p.add_argument("--out", required=True)
    p.add_argument("--tool", default="peft"); p.add_argument("--method", default="lora-sft-benign")
    p.add_argument("--base-model-dir", default=None); p.set_defaults(f=cmd_declare)

    p = sp.add_parser("component-null", help="fit the intra-weight (tensor-group) null from honest derivations")
    p.add_argument("--ref", required=True); p.add_argument("--work", required=True)
    p.add_argument("--prefix", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_component_null)

    p = sp.add_parser("policy", help="write a policy-override JSON file for attest --policy @file")
    p.add_argument("--out", required=True); p.add_argument("--set", action="append", default=[], metavar="field=json")
    p.set_defaults(f=cmd_policy)

    p = sp.add_parser("report", help="plain-text summary of one certificate")
    p.add_argument("attestation"); p.add_argument("--out", default=None); p.set_defaults(f=cmd_report)

    p = sp.add_parser("tamper", help="flip one byte in a COPY of a bundle (invalidation test)")
    p.add_argument("--src", required=True); p.add_argument("--out", required=True); p.set_defaults(f=cmd_tamper)

    a = ap.parse_args(argv)
    if a.cmd == "honest-rtn" and a.prefix is None:
        a.prefix = f"cal{a.bits}"
    rc = a.f(a)
    return rc or 0


if __name__ == "__main__":
    raise SystemExit(main())
