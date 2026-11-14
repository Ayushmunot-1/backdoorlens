"""Build paper/PPT tables and figures from results.json ONLY. No number is typed by hand."""
import argparse, json, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
R = json.loads(Path(a.results).read_text())
out = Path(a.out); (out / "figures").mkdir(parents=True, exist_ok=True); (out / "tables").mkdir(exist_ok=True)
S = R["summary"]
STRAT = [s for s in ("guided", "random", "benchmark") if s in S["honest_false_alarm"]]
f3 = lambda x: "--" if x is None else f"{x:.3g}"
pct = lambda x: f"{100*x:.0f}\\%"
ci = lambda c: f"[{c[0]:.2f}, {c[1]:.2f}]"


def tex_table(name, header, rows, caption, label, colspec=None):
    colspec = colspec or "l" + "c" * (len(header) - 1)
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{%s}" % caption, r"\label{%s}" % label, r"\footnotesize",
             r"\begin{tabular}{%s}" % colspec, r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    lines += [" & ".join(str(c) for c in r) + r" \\" for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (out / "tables" / f"{name}.tex").write_text("\n".join(lines))
    (out / "tables" / f"{name}.json").write_text(json.dumps({"header": header, "rows": rows}))


esc = lambda s: str(s).replace("_", r"\_")
# T1: conventional checks vs BackdoorLens (guided)
rows = []
for n, e in R["E1_conventional"].items():
    if e["class"] != "tampered":
        continue
    g = [x for x in R["audits"] if x["artifact"] == n and x["strategy"] == "guided"]
    det = np.mean([x["detected"] for x in g])
    rows.append([esc(n), "pass", e["serialization_scan"][:4], e["template_static_scan"][:4],
                 f"{e['perplexity_check'].lower()} ({100*e['perplexity_relative_change']:.1f}\\%)", pct(det)])
tex_table("t1_conventional", ["Tampered artifact", "Hash/sig", "Serial.", "Template scan", "PPL check (rel.\\ $\\Delta$NLL)", "BackdoorLens detect"], rows,
          "Standard pre-deployment checks versus BackdoorLens on controlled tampered artifacts (guided search; detect = SUSPICIOUS or BEHAVIORALLY\\_DIVERGENT).", "tab:conventional", "lccccc")
# T2: honest false alarm + LOO
rows = []
for s in STRAT:
    h = S["honest_false_alarm"][s]
    rows.append([s, h["n"], pct(h["fpr"]), ci(h["ci"])])
tex_table("t2_fpr", ["Strategy", "Honest audits", "False-alarm rate", "95\\% CI (Clopper--Pearson)"], rows,
          "False-alarm rate on held-out honest derivations (test split). Audits of one derivation are correlated, so the CI is optimistic.", "tab:fpr", "lccc")
# T3: matched-budget detection
rows = []
for art, d in S["tamper_detection"].items():
    r = [esc(art)]
    for s in STRAT:
        x = d[s]
        r.append(f"{pct(x['detect_rate'])} (n={x['n_runs']})")
    for s in STRAT:
        x = d[s]
        r.append("--" if x["median_first_hit_eval"] is None else f"{x['median_first_hit_eval']:.0f}")
    rows.append(r)
tex_table("t3_search", ["Artifact"] + [f"Detect {s}" for s in STRAT] + [f"1st hit {s}" for s in STRAT], rows,
          "Detection rate and median first-hit evaluation index at an identical paired-case budget. ``1st hit'' = running-best ratio exceeds the effect-size guard.", "tab:search", "l" + "c" * (2 * len(STRAT)))
# T4: calibration ablation
rows = []
for k, v in R["E4_calibration_ablation"]["per_family"].items():
    fam, rule = k.split(":")
    rows.append([fam, esc(rule), pct(v["fpr"]), v["n"], ci(v["ci"])])
for fam, v in R["E4_calibration_ablation"].get("pooled_replicate_search_rule_fpr", {}).items():
    rows.append([fam, "pooled\\_replicate\\_search\\_p", pct(v["fpr"]), v["n"], ci(v["ci"])])
tex_table("t4_calibration", ["Family", "Threshold rule", "FPR", "n", "95\\% CI"], rows,
          "Calibration ablation: global thresholds versus transformation-matched null on honest derivations.", "tab:calibration", "llccc")
# T5: output-only vs logit
rows = []
for n, e in R["E5_output_vs_logit"].items():
    rows.append([esc(n), e["class"], "flag" if e["output_only_flag"] else "ok", "flag" if e["logit_audit_flag"] else "ok"])
tex_table("t5_output_vs_logit", ["Artifact", "Class", "Output-only", "BackdoorLens"], rows,
          "Output-only comparison (generated-text drift above largest honest drift) versus white-box audit.", "tab:output", "llcc")
# T6: invalidation
e8 = R["E8_invalidation"]
rows = [[esc(c["component"]), c["state"], "yes" if c["expected_invalidated"] else "no", f"{1000*c['latency_s']:.1f}"] for c in e8["cases"]]
tex_table("t6_invalidation", ["Modified component", "Certificate state", "Should invalidate", "Check time (ms)"], rows,
          f"Lifecycle invalidation: {e8['tp']}/{e8['n_expected']} in-scope changes invalidated, {e8['fp']} false invalidations.", "tab:invalidation", "lccc")

# ---- figures --------------------------------------------------------------------------
plt.rcParams.update({"font.size": 9, "figure.dpi": 150})
colors = {"guided": "#1f6feb", "random": "#d29922", "benchmark": "#8b949e"}
tam = list(S["tamper_detection"].keys())
fig, ax = plt.subplots(figsize=(6.4, 2.8))
w = 0.8 / len(STRAT)
for i, s in enumerate(STRAT):
    y = [S["tamper_detection"][t][s]["detect_rate"] for t in tam]
    lo = [S["tamper_detection"][t][s]["detect_rate"] - S["tamper_detection"][t][s]["ci"][0] for t in tam]
    hi = [S["tamper_detection"][t][s]["ci"][1] - S["tamper_detection"][t][s]["detect_rate"] for t in tam]
    ax.bar(np.arange(len(tam)) + i * w, y, w, yerr=[lo, hi], label=s, color=colors[s], capsize=2)
ax.set_xticks(np.arange(len(tam)) + w * (len(STRAT) - 1) / 2); ax.set_xticklabels(tam, rotation=30, ha="right")
ax.set_ylabel("detection rate"); ax.set_ylim(0, 1.05); ax.legend(frameon=False, ncol=3)
ax.set_title("Detection at identical paired-case budget (error bars: 95% CI)")
fig.tight_layout(); fig.savefig(out / "figures" / "fig_detection.pdf"); fig.savefig(out / "figures" / "fig_detection.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
for art in [t for t in tam if t.startswith("trigger_easy") or t.startswith("canary_2")][:2]:
    for s in STRAT:
        rows_ = [x for x in R["audits"] if x["artifact"] == art and x["strategy"] == s]
        if not rows_: continue
        c = np.median([x["curve_ratio"] for x in rows_], axis=0)
        ax.plot(np.linspace(1, R["config"]["budget"], len(c)), c, color=colors[s], ls="-" if "trigger" in art else ":", label=f"{s}/{art}")
ax.set_yscale("log"); ax.set_xlabel("paired evaluations"); ax.set_ylabel("best ratio (JSD / level)")
ax.legend(fontsize=5, frameon=False); fig.tight_layout(); fig.savefig(out / "figures" / "fig_curve.pdf"); fig.savefig(out / "figures" / "fig_curve.png"); plt.close(fig)

fig, ax = plt.subplots(figsize=(3.4, 2.6))
for bits, c in ((4, "#1f6feb"), (8, "#2da44e")):
    fp = [x["level"] for x in R["audits"] if x["class"] == "honest" and x["bits"] == bits and x["strategy"] == "guided"]
    ax.hist(np.log10(fp), bins=15, alpha=.6, color=c, label=f"honest INT{bits}")
ax.set_xlabel("log10 median per-prompt JSD (level)"); ax.set_ylabel("audits"); ax.legend(frameon=False)
ax.set_title("Benign drift differs by family"); fig.tight_layout(); fig.savefig(out / "figures" / "fig_levels.pdf"); fig.savefig(out / "figures" / "fig_levels.png"); plt.close(fig)

# macros for the paper text (numbers come from results only)
M = {}
M["nHonestAudits"] = S["honest_false_alarm"]["guided"]["n"]
M["fprGuided"] = f"{100*S['honest_false_alarm']['guided']['fpr']:.1f}"
M["fprGuidedHi"] = f"{100*S['honest_false_alarm']['guided']['ci'][1]:.1f}"
M["looN"] = sum(v["n"] for v in R["LOO_calibration_false_alarm"].values())
M["looFA"] = sum(round(v["fpr"] * v["n"]) for v in R["LOO_calibration_false_alarm"].values())
M["invTP"], M["invN"], M["invFP"] = e8["tp"], e8["n_expected"], e8["fp"]
M["totalSeconds"] = R["total_seconds"]
M["backend"] = R["config"]["backend"]
M["budget"] = R["config"]["budget"]
M["suiteId"] = R["suite_id"]
conv_pass = sum(1 for e in R["E1_conventional"].values() if e["class"] == "tampered" and e["perplexity_check"] == "PASS" and e["template_static_scan"].startswith("PASS"))
M["convPass"], M["convN"] = conv_pass, sum(1 for e in R["E1_conventional"].values() if e["class"] == "tampered")
(out / "macros.tex").write_text("\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in M.items()))
(out / "macros.json").write_text(json.dumps(M, indent=1))
print("report written to", out, M)

# ---- auto-generated findings: every sentence is computed from results.json (negatives included) -------
F = []
syn = bool(R["config"].get("synthetic", False))
hf = S["honest_false_alarm"]
for s_ in STRAT:
    h = hf[s_]
    F.append(f"{s_}: {round(h['fpr']*h['n'])} of {h['n']} honest test audits were flagged (rate {100*h['fpr']:.1f}%, 95% CI {h['ci'][0]*100:.1f}-{h['ci'][1]*100:.1f}%).")
better = tie = worse = 0
for art, d in S["tamper_detection"].items():
    if "guided" in d and "random" in d:
        g, r_ = d["guided"]["detect_rate"], d["random"]["detect_rate"]
        better += g > r_; worse += g < r_; tie += g == r_
F.append(f"Guided search detected strictly more runs than random search on {better} of {better+tie+worse} tampered artifacts, tied on {tie}, and fewer on {worse}.")
for art, d in S["tamper_detection"].items():
    F.append(f"{art}: " + "; ".join(f"{s_} {d[s_]['detect_rate']*100:.0f}% (n={d[s_]['n_runs']})" for s_ in STRAT if s_ in d) + ".")
missed = [a for a, d in S["tamper_detection"].items() if all(d[s_]["detect_rate"] == 0 for s_ in STRAT if s_ in d)]
F.append("Artifacts not detected by any strategy: " + (", ".join(missed) if missed else "none") + ".")
F.append(f"Standard checks (hash, serialization, template scan, perplexity) all passed on {M['convPass']} of {M['convN']} tampered artifacts.")
F.append(f"Scope invalidation: {M['invTP']} of {M['invN']} in-scope modifications invalidated the certificate; {M['invFP']} false invalidations on out-of-scope changes.")
inc = S.get("n_inconclusive_honest", 0)
F.append(f"{inc} honest audits ended INCONCLUSIVE (these are neither false alarms nor passes).")
(out / "findings.json").write_text(json.dumps({"synthetic": syn, "findings": F}, indent=1))
(out / "findings.tex").write_text("\\begin{itemize}\n" + "\n".join("\\item " + f.replace("%", "\\%").replace("_", "\\_") for f in F) + "\n\\end{itemize}")
(out / "macros.tex").write_text((out / "macros.tex").read_text() + f"\n\\newcommand{{\\syntheticrun}}{{{1 if syn else 0}}}")
print("findings:", len(F))
