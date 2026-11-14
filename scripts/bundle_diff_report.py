"""Print the FULL per-field bundle-differential breakdown for a reference/derived pair -- not just
the counts that end up in an attestation's evidence.bundle (n_behavior_diffs / n_undeclared_behavior_diffs).

Why this exists: declare_and_reaudit.py's re-audit of the real adapter came back SUSPICIOUS/QUARANTINE
again, but for a DIFFERENT reason than the first pass -- this time every statistical test actually ran
and all three (p_level, p_spike, p_search) were clearly non-significant (p ~ 0.68-1.0, and the artifact's
own level/spike ratios were below every honest adapter's). The verdict instead came from
"17 undeclared behavior-relevant bundle difference(s)" -- decision.py's declared-scope-compliance check
(bundle_diff.py), which is a SEPARATE mechanism from the statistical tests: it flags any config/tokenizer/
generation/runtime field that differs between reference and derived bundle but was not listed in
declared_changes, specifically so an attacker can't claim "only adapters changed" while secretly also
editing the tokenizer or chat template.

The RIGHT fix is not to guess which fields to add to declared_changes for the real adapter alone --
that risks quietly declaring away a real signal. Instead this script lets you compare the real adapter's
undeclared diffs against the SAME diff run on one of the 6 honest, benign LoRA adapters you already
trained (results\\real_world_adapter\\honest_adapters\\honest_lora_0 etc.) -- if a field differs for
EVERY honest adapter of this family too (e.g. because peft's adapter-only save genuinely never writes
its own config.json/generation_config.json, so those fields compare "missing" vs the reference's values
on every single honest instance), that is a structural, family-wide consequence of the declared
transformation itself, not evidence specific to the real artifact -- and only then is it legitimate to
add it to declared_changes.

Run (no training, no model changes -- this only loads tokenizer/config/json files, fast):
    python scripts\\bundle_diff_report.py --ref models\\SmolLM-135M --bundle models\\real_adapter --declared adapters
    python scripts\\bundle_diff_report.py --ref models\\SmolLM-135M --bundle results\\real_world_adapter\\honest_adapters\\honest_lora_0 --declared adapters
    python scripts\\bundle_diff_report.py --ref models\\SmolLM-135M --bundle results\\real_world_adapter\\honest_adapters\\honest_lora_1 --declared adapters
    ... (repeat for honest_lora_2..5)

Send back the full printed output for the real adapter AND for all 6 honest ones -- that is what lets us
build a declared_changes list that is verified against the honest family, not guessed from the one
artifact under test.
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backdoorlens.bundle_diff import diff_bundles

ap = argparse.ArgumentParser()
ap.add_argument("--ref", required=True)
ap.add_argument("--bundle", required=True, help="the derived bundle to diff against --ref")
ap.add_argument("--declared", nargs="*", default=[], help="declared_changes components, e.g. adapters")
a = ap.parse_args()

rep = diff_bundles(a.ref, a.bundle, a.declared)
beh = rep.behavior_diffs
und = rep.undeclared_behavior_diffs

print(f"\n=== bundle diff: {a.bundle}  vs ref: {a.ref}  (declared_changes={a.declared}) ===")
print(f"n_behavior_diffs={len(beh)}  n_undeclared_behavior_diffs={len(und)}")
print(f"n_rendered_mismatches={len(rep.rendered_mismatches)}  n_render_errors={len(rep.render_errors)}  "
      f"template_flags={rep.template_flags}")
print("\nbehavior-severity differences (component.field: reference -> derived   [declared?]):")
for d in beh:
    tag = "DECLARED" if d.declared else "UNDECLARED"
    rv = repr(d.reference)[:80]
    dv_ = repr(d.derived)[:80]
    print(f"  [{tag:11s}] {d.component}.{d.field}: {rv} -> {dv_}")
if rep.rendered_mismatches:
    print("\nrendered_mismatches (first 3):")
    for m in rep.rendered_mismatches[:3]:
        print(f"  case={m['case']!r}\n    ref: {m['reference']!r}\n    der: {m['derived']!r}")
