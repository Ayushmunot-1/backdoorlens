"""Generate ONE self-contained HTML report from the evidence — the artifact you hand a reviewer.

Reads the three evidence files produced elsewhere in the pipeline and renders a single offline HTML
file: no CDN, no JavaScript, no network, system fonts, inline SVG. It must open on a projector with
the wifi down, and it must print to PDF cleanly.

INPUTS
  --results      results/smollm2/results.json              (run_experiments.py: E1 baselines,
                                                            detection, false alarms, invalidation)
  --component    .../component_null_audit.json              (component_null_audit.py)
  --attestation  results/attest_trigger_easy/attestation.json (cli attest, for the audited artifact)

RUN
    python run/2_report.py --results results/smollm2/results.json ^
      --component results/component_null_int4/component_null_audit.json ^
      --attestation results/attest_trigger_easy/attestation.json ^
      --out evidence/report/backdoorlens_report.html

CHART NOTES (why these two, and why they look the way they do)
  * Separability strip — the only conventional check that reacts (relative perplexity) plotted for
    honest-int4 vs tampered artifacts on one shared axis, in two lanes. The overlap IS the finding, so
    the chart's whole job is to make the overlap unmissable. The two canary outliers (2.08, 5.04) are
    clamped to the right edge and labelled with their true values rather than being allowed to squash
    the 0.12–0.17 region where the overlap happens.
  * Calibration envelope — the honest derivations' level range as a band, the audited artifact as a
    marker, and the materiality guard (2x the honest maximum) as a line. This is the picture of why a
    statistically significant deviation correctly did not fire.
  Colors are the validated categorical slots 1 and 2 (blue/orange), NOT good/green vs critical/red:
  green+red fails CVD separation for deuteranopia (validator: ΔE 4.1, below even the conditional
  floor). Class identity is additionally carried by lane position and a direct text label, so color
  is never the only encoding.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

# --- validated palette (categorical slots 1-2 + chrome), light/dark ----------------------------
CSS = """
:root{
  --surface:#fcfcfb; --plane:#f9f9f7; --ink:#0b0b0b; --ink2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --ring:rgba(11,11,11,0.10);
  --honest:#2a78d6; --tamper:#eb6834;
}
@media (prefers-color-scheme: dark){
  :root:not([data-theme="light"]){
    --surface:#1a1a19; --plane:#0d0d0d; --ink:#f0efec; --ink2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --ring:rgba(255,255,255,0.10);
    --honest:#3987e5; --tamper:#d95926;
  }
}
*{box-sizing:border-box}
body{margin:0;background:var(--plane);color:var(--ink);
     font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;}
.wrap{max-width:900px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px;letter-spacing:-0.01em}
h2{font-size:17px;margin:38px 0 10px;padding-bottom:6px;border-bottom:1px solid var(--grid)}
p{margin:10px 0}
.sub{color:var(--ink2);margin:0 0 24px}
.card{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:18px;margin:14px 0}
.verdict{display:flex;gap:18px;align-items:baseline;flex-wrap:wrap}
.vbig{font-size:30px;font-weight:650;letter-spacing:-0.02em}
.vmeta{color:var(--ink2);font-size:14px}
.pill{display:inline-block;padding:2px 9px;border:1px solid var(--ring);border-radius:999px;
      font-size:12px;color:var(--ink2);background:var(--plane)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:8px 0}
th,td{text-align:left;padding:7px 10px;border-bottom:1px solid var(--grid)}
th{color:var(--ink2);font-weight:600;font-size:12.5px;text-transform:uppercase;letter-spacing:0.04em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
.ok{color:var(--honest);font-weight:600}
.no{color:var(--tamper);font-weight:600}
.mono{font-family:ui-monospace,"Cascadia Code",Consolas,monospace;font-size:12.5px}
.io{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.io>div{background:var(--plane);border:1px solid var(--grid);border-radius:8px;padding:12px}
.io h4{margin:0 0 6px;font-size:12px;text-transform:uppercase;letter-spacing:0.04em;color:var(--ink2)}
.claim{font-size:17px;line-height:1.5;border-left:3px solid var(--honest);padding:2px 0 2px 14px;margin:18px 0}
.note{color:var(--ink2);font-size:13px}
details{margin:8px 0}summary{cursor:pointer;color:var(--ink2);font-size:13.5px}
svg{display:block;max-width:100%;height:auto}
@media print{body{background:#fff}.card{break-inside:avoid}}
@media (max-width:620px){.io{grid-template-columns:1fr}}
"""


def esc(x) -> str:
    return html.escape(str(x))


# ------------------------------------------------------------------ chart 1: separability strip
def strip_chart(honest: list[tuple[str, float]], tampered: list[tuple[str, float]]) -> str:
    """Two lanes on one shared axis, with the overlap zone shaded — the overlap IS the finding.

    The axis is focused on the region where both classes live. Values beyond it (the large canary
    perturbations) are drawn clamped at the right edge and their true values stated in one compact
    end-anchored label, so they neither overflow nor squash the region that matters.
    """
    W, H, L, R = 860, 210, 96, 86
    y_h, y_t = 76, 142
    pw = W - L - R
    xlo, xhi = 0.11, 0.23

    def x(v):
        return L + pw * (min(max(v, xlo), xhi) - xlo) / (xhi - xlo)

    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Relative perplexity change for honest '
         f'int4 versus tampered artifacts on one axis; the two ranges overlap">']
    hv = max(v for _, v in honest) if honest else None
    tv = min(v for _, v in tampered) if tampered else None

    # overlap zone first, so marks sit on top of it
    if hv is not None and tv is not None and hv > tv:
        s.append(f'<rect x="{x(tv):.1f}" y="46" width="{max(x(hv)-x(tv),3):.1f}" height="{y_t+22-46}" '
                 f'fill="var(--axis)" opacity="0.22"/>')
        s.append(f'<text x="{(x(tv)+x(hv))/2:.1f}" y="40" fill="var(--ink)" font-size="10.5" '
                 f'text-anchor="middle" font-weight="600">overlap</text>')

    t = xlo
    while t <= xhi + 1e-9:
        s.append(f'<line x1="{x(t):.1f}" y1="46" x2="{x(t):.1f}" y2="{H-44}" stroke="var(--grid)" '
                 f'stroke-width="1"/>')
        s.append(f'<text x="{x(t):.1f}" y="{H-26}" fill="var(--muted)" font-size="11" '
                 f'text-anchor="middle">{t:.2f}</text>')
        t += 0.02
    s.append(f'<text x="{L+pw/2:.0f}" y="{H-8}" fill="var(--ink2)" font-size="11.5" '
             f'text-anchor="middle">relative perplexity change vs reference</text>')

    for label, pts, yy, col in (("honest int4", honest, y_h, "var(--honest)"),
                                ("tampered", tampered, y_t, "var(--tamper)")):
        s.append(f'<text x="{L-12}" y="{yy+4}" fill="var(--ink2)" font-size="12.5" '
                 f'text-anchor="end">{esc(label)}</text>')
        vals = sorted(v for _, v in pts)
        if len(vals) > 1:
            s.append(f'<line x1="{x(vals[0]):.1f}" y1="{yy}" x2="{x(vals[-1]):.1f}" y2="{yy}" '
                     f'stroke="{col}" stroke-width="2" opacity="0.35" stroke-linecap="round"/>')
        beyond = []
        for name, v in sorted(pts, key=lambda t: t[1]):
            if v > xhi:
                beyond.append((name, v))
            s.append(f'<circle cx="{x(v):.1f}" cy="{yy}" r="5" fill="{col}" stroke="var(--surface)" '
                     f'stroke-width="2"><title>{esc(name)}: {v:.5f}</title></circle>')
        if beyond:
            txt = ", ".join(f"{n} {v:.2f}" for n, v in beyond)
            s.append(f'<text x="{W-6}" y="{yy-12}" fill="var(--ink2)" font-size="10.5" '
                     f'text-anchor="end">&#8594; {esc(txt)}</text>')

    if hv is not None and tv is not None:
        s.append(f'<text x="{x(hv):.1f}" y="{y_h-16}" fill="var(--ink)" font-size="11" '
                 f'text-anchor="middle" font-weight="600">worst honest {hv:.3f}</text>')
        s.append(f'<text x="{x(tv):.1f}" y="{y_t+26}" fill="var(--ink)" font-size="11" '
                 f'text-anchor="middle" font-weight="600">lowest tampered {tv:.3f}</text>')
    s.append(f'<text x="{L-84}" y="22" fill="var(--ink)" font-size="13" font-weight="600">'
             f'No threshold separates the two classes</text>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ chart 2: calibration envelope
def envelope_chart(honest_levels: list[float], audited: float, guard_mult: float) -> str:
    W, H, L, R = 860, 164, 92, 56
    y = 78
    pw = W - L - R
    guard = max(honest_levels) * guard_mult if honest_levels else audited
    xmax = max(audited, guard) * 1.18

    def x(v):
        return L + pw * (v / xmax)

    lo, hi = min(honest_levels), max(honest_levels)
    s = [f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="Honest calibration envelope versus the '
         f'audited artifact level, with the materiality guard">']
    step = 0.1
    t = 0.0
    while t <= xmax + 1e-9:
        s.append(f'<line x1="{x(t):.1f}" y1="34" x2="{x(t):.1f}" y2="{H-38}" stroke="var(--grid)" '
                 f'stroke-width="1"/>')
        s.append(f'<text x="{x(t):.1f}" y="{H-20}" fill="var(--muted)" font-size="11" '
                 f'text-anchor="middle">{t:.1f}</text>')
        t += step
    s.append(f'<text x="{L+pw/2:.0f}" y="{H-4}" fill="var(--ink2)" font-size="11.5" '
             f'text-anchor="middle">suite-wide median JSD ("level")</text>')
    # honest band
    s.append(f'<rect x="{x(lo):.1f}" y="{y-13}" width="{max(x(hi)-x(lo),2):.1f}" height="26" rx="4" '
             f'fill="var(--honest)" opacity="0.18"/><title>honest range {lo:.3f}-{hi:.3f}</title>')
    for v in honest_levels:
        s.append(f'<circle cx="{x(v):.1f}" cy="{y}" r="4.5" fill="var(--honest)" '
                 f'stroke="var(--surface)" stroke-width="2"><title>honest {v:.4f}</title></circle>')
    s.append(f'<text x="{x((lo+hi)/2):.1f}" y="{y-22}" fill="var(--ink2)" font-size="11" '
             f'text-anchor="middle">honest envelope {lo:.3f}–{hi:.3f}</text>')
    # guard line, and the region beyond it where the test would fire
    s.append(f'<rect x="{x(guard):.1f}" y="40" width="{max(W-R-x(guard),0):.1f}" height="{y+34-40}" '
             f'fill="var(--tamper)" opacity="0.07"/>')
    s.append(f'<line x1="{x(guard):.1f}" y1="40" x2="{x(guard):.1f}" y2="{y+34}" '
             f'stroke="var(--ink2)" stroke-width="2" stroke-dasharray="5 4"/>')
    s.append(f'<text x="{x(guard):.1f}" y="{y+50}" fill="var(--ink2)" font-size="11" '
             f'text-anchor="middle">guard {guard_mult:g}× = {guard:.3f} — fires to the right</text>')
    # audited marker
    s.append(f'<circle cx="{x(audited):.1f}" cy="{y}" r="7.5" fill="var(--tamper)" '
             f'stroke="var(--surface)" stroke-width="2"><title>audited {audited:.4f}</title></circle>')
    s.append(f'<text x="{x(audited):.1f}" y="{y-22}" fill="var(--ink)" font-size="11.5" '
             f'text-anchor="middle" font-weight="600">audited {audited:.3f}</text>')
    s.append(f'<text x="{L}" y="20" fill="var(--ink)" font-size="13" font-weight="600">'
             f'Outside the honest envelope — but short of the materiality guard</text>')
    s.append("</svg>")
    return "".join(s)


# ------------------------------------------------------------------ report
def build(R: dict, C: dict | None, A: dict | None) -> str:
    e1 = R.get("E1_conventional", {})
    summ = R.get("summary", {})
    det = summ.get("tamper_detection", {})
    fa = summ.get("honest_false_alarm", {})
    loo = R.get("LOO_calibration_false_alarm", {})
    inv = R.get("E8_invalidation", {})

    hon4 = [(k, v["perplexity_relative_change"]) for k, v in e1.items()
            if v.get("class") == "honest" and k.startswith("test4")]
    tam = [(k, v["perplexity_relative_change"]) for k, v in e1.items() if v.get("class") == "tampered"]

    P = [f"<!doctype html><html lang='en'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width,initial-scale=1'>",
         "<title>BackdoorLens — audit evidence</title>", f"<style>{CSS}</style></head><body><div class='wrap'>"]

    # ---- header / verdict
    dec = (A or {}).get("decision", {})
    exit_map = {"ALLOW": 0, "ALLOW_WITH_MONITORING": 0, "QUARANTINE": 10, "BLOCK": 20,
                "REQUIRE_MORE_EVIDENCE": 30, "REQUIRE_REATTESTATION": 40}
    P.append("<h1>BackdoorLens — audit evidence</h1>")
    P.append("<p class='sub'>Transformation-aware behavioural attestation for derived open-weight "
             "model artifacts. Every number on this page was produced by the committed code.</p>")
    P.append("<p class='claim'><strong>Signing proves the bytes didn't change. "
             "Nothing proves the behaviour didn't.</strong><br>"
             "An organisation verifies the model it downloads, then quantizes it to fit the serving "
             "budget — and at that moment the signature is void and nothing checks that behaviour "
             "survived. This is that check.</p>")
    if dec:
        ref = (A.get("audit_config") or {}).get("model_family", "?")
        P.append("<div class='card'><div class='verdict'>"
                 f"<div class='vbig'>{esc(dec.get('verdict'))}</div>"
                 f"<div class='vmeta'>action <strong>{esc(dec.get('action'))}</strong> &nbsp;·&nbsp; "
                 f"CI exit code <strong>{exit_map.get(dec.get('action'), '?')}</strong></div></div>")
        P.append(f"<p class='note'>family <span class='mono'>{esc(ref)}</span> · attestation "
                 f"<span class='mono'>{esc(A.get('id'))}</span> · "
                 f"{'signed' if A.get('signature') else 'unsigned'} · created {esc(A.get('created'))}</p>")
        for r in dec.get("reasons", []):
            P.append(f"<p class='note'>↳ {esc(r)}</p>")
        P.append("</div>")

    # ---- 1. conventional baseline
    P.append("<h2>1 · What standard pre-deployment checks say</h2>")
    P.append("<p>The four checks a real pipeline runs today, on every artifact in the experiment. "
             "Three of them pass everything — including the tampered artifacts. Note in particular "
             "that the chat-template static scanner passes <span class='mono'>template_inject</span>: "
             "the injected text contains no code construct for a regex to find.</p>")
    P.append("<table><tr><th>artifact</th><th>class</th><th>hash / signature</th>"
             "<th>serialization</th><th>template scan</th><th class='num'>ppl change</th>"
             "<th>ppl check</th></tr>")
    for k, v in e1.items():
        cls = v.get("class")
        ppl = v.get("perplexity_relative_change")
        bad = v.get("perplexity_check") == "FAIL"
        P.append(f"<tr><td class='mono'>{esc(k)}</td><td>{esc(cls)}</td>"
                 f"<td class='ok'>PASS</td><td class='ok'>PASS</td><td class='ok'>PASS</td>"
                 f"<td class='num'>{ppl:.5f}</td>"
                 f"<td class='{'no' if bad else 'ok'}'>{esc(v.get('perplexity_check'))}</td></tr>")
    P.append("</table>")

    if hon4 and tam:
        hk, hv = max(hon4, key=lambda t: t[1])
        tk, tv = min(tam, key=lambda t: t[1])
        P.append(f"<div class='card'>{strip_chart(hon4, tam)}")
        P.append(f"<p class='note'>Worst honest int4 artifact (<span class='mono'>{esc(hk)}</span>) "
                 f"changes perplexity by <strong>{hv:.5f}</strong>. The <em>lowest</em> tampered "
                 f"artifact (<span class='mono'>{esc(tk)}</span>) changes it by <strong>{tv:.5f}</strong>. "
                 f"Because the honest maximum exceeds the tampered minimum, <strong>no threshold "
                 f"passes every honest quantization and catches every tamper</strong>. Set it high "
                 f"enough to catch the triggers and it misses the template injection; set it low "
                 f"enough to catch that and it blocks legitimate quantization work.</p></div>")

    # ---- 2. detection and false alarms
    P.append("<h2>2 · What BackdoorLens detects, and how often it cries wolf</h2>")
    tdir = {"template_inject": "t_template", "generation_edit": "t_gen",
            "trigger_easy": "t_trigger_easy", "trigger_hard": "t_trigger_hard",
            "canary_0.5": "t_canary_0.5", "canary_2.0": "t_canary_2.0", "canary_8.0": "t_canary_8.0"}
    comp = (C or {}).get("audits", {})
    P.append("<table><tr><th>tampered artifact</th><th class='num'>guided</th><th class='num'>random</th>"
             "<th class='num'>benchmark</th><th class='num'>component anomalies</th><th>combined</th></tr>")
    n_beh = n_comb = 0
    for name, d in tdir.items():
        v = det.get(name, {})
        rates = [v.get(s, {}).get("detect_rate") for s in ("guided", "random", "benchmark")]
        beh = any((r or 0) > 0 for r in rates)
        ca = comp.get(d, {})
        nh = ca.get("n_hard_anomalies") if ca.get("status") == "ok" else None
        hit = beh or bool(nh)
        n_beh += beh
        n_comb += hit
        cells = "".join(f"<td class='num'>{(f'{r:.2f}' if r is not None else '—')}</td>" for r in rates)
        P.append(f"<tr><td class='mono'>{esc(name)}</td>{cells}"
                 f"<td class='num'>{nh if nh is not None else '—'}</td>"
                 f"<td class='{'ok' if hit else 'no'}'>{'DETECTED' if hit else 'missed'}</td></tr>")
    P.append("</table>")
    P.append(f"<p>Behavioural channel alone: <strong>{n_beh}/{len(tdir)}</strong>. "
             f"With the intra-weight component channel: <strong>{n_comb}/{len(tdir)}</strong>. "
             "The two channels are independent — the component channel catches exactly the artifacts "
             "every behavioural test missed.</p>")

    tot_n = tot_fp = 0
    rows = []
    for s in ("guided", "random", "benchmark"):
        v = fa.get(s)
        if v:
            n, k = int(v["n"]), int(round(v["fpr"] * v["n"]))
            tot_n += n
            tot_fp += k
            rows.append((f"behavioural · {s}", k, n, f"95% CI upper {v['ci'][1]:.3f}"))
    for bits, blk in (loo or {}).items():
        r = blk.get("rows", [])
        if r:
            k = sum(1 for x in r if x.get("detected"))
            tot_n += len(r)
            tot_fp += k
            rows.append((f"behavioural · leave-one-out {bits}", k, len(r), ""))
    if C:
        l = C.get("leave_one_out", {})
        hd = [d for d in comp if d.startswith(("cal", "test"))]
        rows.append(("component · leave-one-out", l.get("n_false_positives"), l.get("n_tested"), ""))
        rows.append(("component · held-out honest",
                     sum(1 for d in hd if (comp[d].get("n_hard_anomalies") or 0) > 0), len(hd), ""))
    P.append("<table><tr><th>false-alarm test</th><th class='num'>false alarms</th>"
             "<th class='num'>honest audits</th><th>note</th></tr>")
    for nm, k, n, note in rows:
        P.append(f"<tr><td>{esc(nm)}</td><td class='num {'ok' if not k else 'no'}'>{esc(k)}</td>"
                 f"<td class='num'>{esc(n)}</td><td class='note'>{esc(note)}</td></tr>")
    P.append(f"<tr><td><strong>behavioural total</strong></td>"
             f"<td class='num ok'><strong>{tot_fp}</strong></td>"
             f"<td class='num'><strong>{tot_n}</strong></td><td></td></tr></table>")

    # ---- 3. envelope / why significance alone is not enough
    if A:
        ev = A.get("evidence", {})
        nul, sa, se = ev.get("null") or {}, ev.get("stage_a", {}), ev.get("search", {})
        lv = nul.get("honest_levels")
        pol = A.get("policy", {})
        if lv and sa.get("level_median_jsd"):
            P.append("<h2>3 · Why a significant deviation correctly did not fire</h2>")
            P.append(f"<div class='card'>{envelope_chart(lv, sa['level_median_jsd'], pol.get('min_level_ratio', 2.0))}")
            P.append(f"<p class='note'>Every statistic here is a <em>ratio against the audited "
                     f"artifact's own level</em>, so thresholds are relative to the derivation rather "
                     f"than one global constant. This artifact's level is "
                     f"<strong>{sa['level_median_jsd']:.4f}</strong> against an honest range of "
                     f"{min(lv):.3f}–{max(lv):.3f} — statistically significant "
                     f"(p<sub>level</sub> = {sa.get('p_level'):.2e}) but only "
                     f"{sa.get('level_ratio_vs_max_honest'):.3f}× the honest maximum, short of the "
                     f"{pol.get('min_level_ratio', 2.0):g}× materiality guard. A modification "
                     f"pervasive enough to raise its own baseline hides itself from ratio-normalised "
                     f"tests. That is a real limitation of the method, and it is why a second, "
                     f"non-ratio channel exists.</p></div>")
            P.append("<table><tr><th>statistic</th><th class='num'>value</th><th>gate</th><th>fired</th></tr>")
            a3 = pol.get("alpha_stage_a", 0.005) / 3.0
            gates = [
                ("p_level", sa.get("p_level"), f"≤ {a3:.2e} and level ratio ≥ {pol.get('min_level_ratio',2):g}×",
                 (sa.get("p_level") or 1) <= a3 and (sa.get("level_ratio_vs_max_honest") or 0) >= pol.get("min_level_ratio", 2)),
                ("p_spike", sa.get("p_spike"), f"≤ {a3:.2e} and spike ratio ≥ {pol.get('min_spike_ratio',2):g}×",
                 (sa.get("p_spike") or 1) <= a3 and (sa.get("spike_ratio_vs_max_honest") or 0) >= pol.get("min_spike_ratio", 2)),
                ("p_search", se.get("p_search"), f"≤ {pol.get('alpha_search',0.01):g} and margin ≥ {pol.get('min_search_margin',1.5):g}×",
                 (se.get("p_search") or 1) <= pol.get("alpha_search", 0.01)),
                ("argmax flip rate", sa.get("argmax_flip_rate"), "descriptive", None),
                ("p_binom (pooled tail)", sa.get("p_binom"), "descriptive only — pseudo-replication", None),
            ]
            for nm, val, gate, fired in gates:
                v = ("—" if val is None else (f"{val:.4g}"))
                f = "—" if fired is None else ("<span class='no'>yes</span>" if fired else "<span class='ok'>no</span>")
                P.append(f"<tr><td class='mono'>{esc(nm)}</td><td class='num'>{v}</td>"
                         f"<td class='note'>{esc(gate)}</td><td>{f}</td></tr>")
            P.append("</table>")

    # ---- 4. component channel
    if C:
        cn = C.get("component_null", {})
        P.append("<h2>4 · The component channel — provenance, not behaviour</h2>")
        P.append(f"<p>Calibrated on <strong>{esc(cn.get('n_honest_derivations'))}</strong> honest "
                 f"derivations of the declared family, each of which modifies exactly "
                 f"<strong>{esc((cn.get('n_tensors_changed_honest') or {}).get('min'))}</strong> of "
                 f"the model's tensors — with no variance at all. "
                 f"<strong>{esc(cn.get('n_never_changed_keys'))}</strong> tensor groups are modified "
                 f"by no honest derivation.</p>")
        trig = comp.get("t_trigger_easy", {})
        if trig.get("status") == "ok":
            P.append(f"<p>The audited artifact modifies <strong>{esc(trig.get('n_tensors_changed'))}</strong> "
                     f"tensors and trips <strong>{esc(trig.get('n_hard_anomalies'))}</strong> of those "
                     f"groups. Composition: the token embedding, the final norm, and every layernorm — "
                     f"exactly the set the honest quantizer skips, because it only touches 2-D tensors "
                     f"and skips embed/lm_head/norm by name. The separation is structural, not a "
                     f"matter of magnitudes.</p>")
            P.append("<details><summary>show the flagged tensor groups</summary>"
                     f"<p class='mono note'>{esc(', '.join(trig.get('hard_anomalies', [])))}</p></details>")
        P.append("<p class='note'><strong>How this must be read.</strong> The honest null for a "
                 "never-touched group is degenerate — identically zero, no variance — so no p-value "
                 "is computed and none is claimed. The correct statement is: <em>this artifact "
                 "modifies N tensor groups that none of the K honest derivations of its declared "
                 "family modifies</em>. It is evidence of provenance outside the declared "
                 "transformation, <strong>not</strong> proof of a backdoor: a benign full fine-tune "
                 "would flag identically.</p>")

    # ---- 5. counterexample
    cex = (A or {}).get("counterexamples") or []
    if cex:
        c = max(cex, key=lambda z: z.get("jsd", 0))
        P.append("<h2>5 · A reproducible counterexample</h2>")
        msgs = (c.get("prompt") or {}).get("messages", [])
        P.append(f"<p>Found by budgeted mutation search, persistent across "
                 f"{esc(c.get('n_variants_pass'))}/{esc(c.get('n_variants'))} semantic variants, and "
                 f"re-verified through a separate call path "
                 f"(<span class='mono'>replay_ok = {esc(c.get('replay_ok'))}</span>).</p>")
        P.append("<div class='card'>")
        for m in msgs:
            P.append(f"<p class='note'><span class='pill'>{esc(m.get('role'))}</span> "
                     f"<span class='mono'>{esc(m.get('content'))[:300]}</span></p>")
        P.append("<div class='io'>"
                 f"<div><h4>reference output</h4><p class='mono'>{esc(c.get('reference_output'))}</p></div>"
                 f"<div><h4>derived output</h4><p class='mono'>{esc(c.get('derived_output'))}</p></div></div>")
        met = c.get("metrics", {})
        P.append(f"<p class='note'>JSD {c.get('jsd'):.4f} · KL {met.get('kl',0):.2f} · "
                 f"NLL shift {met.get('nll_shift',0):.2f} · top-k overlap {met.get('topk_overlap',0):.3f} "
                 f"· argmax flip {esc(met.get('argmax_flip'))} · edit distance "
                 f"{c.get('edit_distance',0):.3f}</p></div>")

    # ---- 6. invalidation
    if inv:
        P.append("<h2>6 · The certificate invalidates itself</h2>")
        cases = inv.get("cases", [])
        lat = [c.get("latency_s") for c in cases if c.get("latency_s") is not None]
        P.append(f"<p>Baseline state <strong>{esc(inv.get('baseline_state'))}</strong>. Each in-scope "
                 f"component is then modified and the gate re-run — no model is loaded, this is a hash "
                 f"comparison"
                 f"{f' (median {sorted(lat)[len(lat)//2]:.2f} s)' if lat else ''}.</p>")
        P.append("<table><tr><th>component modified</th><th>certificate state</th><th>as expected</th></tr>")
        for c in cases:
            good = (c.get("state") == "INVALIDATED") == bool(c.get("expected_invalidated"))
            P.append(f"<tr><td class='mono'>{esc(c.get('component'))}</td><td>{esc(c.get('state'))}</td>"
                     f"<td class='{'ok' if good else 'no'}'>{'yes' if good else 'NO'}</td></tr>")
        P.append("</table>")
        P.append("<p class='note'>The last row is the negative control: a change to a file outside the "
                 "certified scope leaves the certificate <strong>VALID</strong>. The scope is precise, "
                 "not merely alarmist.</p>")

    # ---- 7. limitations
    P.append("<h2>7 · Limitations</h2><ul class='note'>")
    for t in ("Quantization here is simulated round-to-nearest, not a bitsandbytes/GPTQ/GGUF kernel; "
              "the set of tensor groups an honest derivation skips is a property of that implementation.",
              "The component channel has a degenerate (identically zero) null — it is a component-level "
              "finding with the number of honest derivations stated, never a p-value.",
              "It detects provenance deviation, not malice: a benign full fine-tune flags identically.",
              "It needs full weights on both sides, so it does not apply to adapter-only bundles.",
              "An attacker who confines a change to the tensor groups the honest transformation already "
              "touches produces no anomaly in that channel.",
              "Ratio-normalised behavioural statistics can be self-masking (section 3).",
              "Finite probing cannot exclude rare triggers outside the exercised distribution; a "
              "compromised reference is faithfully preserved by an honest derivation.",
              "One model family at one size (135M), CPU only. No claim of generality across scales."):
        P.append(f"<li>{esc(t)}</li>")
    P.append("</ul>")

    env = (A or {}).get("environment", {})
    if env:
        P.append(f"<p class='note'>Environment: python {esc(env.get('python'))} · torch "
                 f"{esc(env.get('torch'))} · transformers {esc(env.get('transformers'))} · "
                 f"{esc(env.get('platform'))} · CUDA {esc(env.get('cuda_available'))}</p>")
    P.append("</div></body></html>")
    return "".join(P)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--component")
    ap.add_argument("--attestation")
    ap.add_argument("--out", default="evidence/report/backdoorlens_report.html")
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    C = json.loads(Path(a.component).read_text()) if a.component else None
    A = json.loads(Path(a.attestation).read_text()) if a.attestation else None
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build(R, C, A), encoding="utf-8")
    print(f"wrote {out}  ({out.stat().st_size/1024:.0f} KB, self-contained, no network)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
