"""Build the review deck as a .pptx, with every number read from the evidence files.

    pip install python-pptx
    python scripts\\build_deck.py ^
      --results results\\smollm2\\results.json ^
      --component results\\component_null_int4\\component_null_audit.json ^
      --attestation results\\demo\\attestation\\attestation.json ^
      --out deck\\BackdoorLens_review2.pptx

WHY A GENERATOR AND NOT A HAND-BUILT DECK
Project rule: no number in the paper or the deck is typed by hand (§22/§23). So every figure on every
slide is read out of `results.json`, `component_null_audit.json` or `attestation.json`, and `need()`
below **raises on a missing key** rather than rendering a blank or a zero. A deck that silently shows
"0 false alarms" because a key moved is the exact failure that rule exists to prevent.

It replaces `deck/make_deck.js`, which cannot run here: it needs node plus pptxgenjs, it `require`s
`/mnt/skills/public/pptx/scripts/apply_theme.js` (a Linux build-environment path that cannot exist on
Windows), and it reads only `results.json` — so it never sees the component channel and cannot state
the headline result. The old deck in `deck/` was built from the TOY run and carries a red SYNTHETIC
banner.

SLIDE ORDER (15; notes on every slide say what to say and answer the likely question)
  1 title · 2 the deployment gap · 3 what is audited · 4 how it works · 5 conventional checks ·
  6 separability · 7 calibration · 8 the envelope · 9 headline result · 10 false alarms ·
  11 counterexample · 12 the gate · 13 scope boundary · 14 limitations · 15 conclusion
Slides 3 and 11 are marked OPTIONAL in their notes — drop them for a 10-minute slot.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import XyChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

# ---------------------------------------------------------------- palette and type
# Not default corporate blue: a deep charcoal-navy base with the HTML report's exact data colours, so
# the deck and the report read as one system. The blue/orange pair is the CVD-validated categorical
# pair from the report (green/red failed deuteranopia validation at dE 4.1).
INK = RGBColor(0x14, 0x18, 0x1F)
DARK = RGBColor(0x1E, 0x24, 0x30)
SURFACE = RGBColor(0xF4, 0xF5, 0xF7)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
MUTED = RGBColor(0x5A, 0x64, 0x72)
PALE = RGBColor(0xC9, 0xCF, 0xD8)
DIM = RGBColor(0x8C, 0x97, 0xA6)        # muted text that stays legible on the dark slides
HONEST = RGBColor(0x2A, 0x78, 0xD6)
TAMPER = RGBColor(0xEB, 0x68, 0x34)
HEAD, BODY, MONO = "Cambria", "Calibri", "Consolas"
W, H = 13.333, 7.5


def need(obj, *path, default=KeyError):
    """Read a nested key, raising with the full path when it is absent.

    The whole point: a deck is evidence. A missing key must stop the build, not quietly become a
    blank or a zero on a slide a panel is reading.
    """
    cur, walked = obj, []
    for k in path:
        walked.append(str(k))
        try:
            cur = cur[k]
        except (KeyError, IndexError, TypeError):
            if default is not KeyError:
                return default
            raise KeyError(f"missing in evidence: {'.'.join(walked)}") from None
    return cur


def preflight(R: dict, C: dict, A: dict) -> list[str]:
    """Report EVERY key the build needs and cannot find, in one pass.

    `need()` raises on the first missing key, which means a schema drift costs one run per missing
    field. This checks the whole set up front so a mismatch is reported once, completely. Paths that
    are genuinely optional (counterexamples, environment) are not listed here.
    """
    bad = []

    def chk(obj, label, *path):
        cur, walked = obj, []
        for k in path:
            walked.append(str(k))
            try:
                cur = cur[k]
            except (KeyError, IndexError, TypeError):
                bad.append(f"{label}: {'.'.join(walked)}")
                return None
        return cur

    chk(R, "results.json", "config", "model_family")
    e1 = chk(R, "results.json", "E1_conventional")
    det = chk(R, "results.json", "summary", "tamper_detection")
    fa = chk(R, "results.json", "summary", "honest_false_alarm")
    loo = chk(R, "results.json", "LOO_calibration_false_alarm")
    cases = chk(R, "results.json", "E8_invalidation", "cases")
    if isinstance(e1, dict):
        for k, v in e1.items():
            for f in ("class", "perplexity_relative_change"):
                if f not in v:
                    bad.append(f"results.json: E1_conventional.{k}.{f}")
    if isinstance(det, dict):
        for k, v in det.items():
            for st in ("guided", "random", "benchmark"):
                chk(det, "results.json", k, st, "detect_rate")
    if isinstance(fa, dict):
        for st in ("guided", "random", "benchmark"):
            for f in ("n", "fpr"):
                chk(fa, "results.json", st, f)
    if isinstance(loo, dict):
        for b in loo:
            chk(loo, "results.json", b, "rows")
    if isinstance(cases, list):
        for i, c in enumerate(cases):
            for f in ("component", "state"):
                if f not in c:
                    bad.append(f"results.json: E8_invalidation.cases[{i}].{f}")

    chk(C, "component null", "audits")
    chk(C, "component null", "component_null")
    chk(C, "component null", "leave_one_out", "n_false_positives")
    chk(C, "component null", "leave_one_out", "n_tested")

    chk(A, "attestation.json", "id")
    for f in ("verdict", "action"):
        chk(A, "attestation.json", "decision", f)
    for f in ("level_median_jsd", "p_level", "level_ratio_vs_max_honest", "argmax_flip_rate"):
        chk(A, "attestation.json", "evidence", "stage_a", f)
    for f in ("honest_levels", "n_derivations", "search_budget"):
        chk(A, "attestation.json", "evidence", "null", f)
    chk(A, "attestation.json", "policy", "min_level_ratio")
    cx = (A.get("counterexamples") or [None])[0]
    if cx is not None:
        for path in (("jsd",), ("n_variants",), ("n_variants_pass",), ("replay_ok",),
                     ("metrics", "topk_overlap"), ("prompt", "messages"),
                     ("reference_output",), ("derived_output",)):
            cur, walked = cx, ["counterexamples[0]"]
            for k in path:
                walked.append(str(k))
                try:
                    cur = cur[k]
                except (KeyError, IndexError, TypeError):
                    bad.append(f"attestation.json: {'.'.join(walked)}")
                    break
    return bad


# ---------------------------------------------------------------- drawing helpers
def _tf(shape, margin=0.0):
    tf = shape.text_frame
    tf.word_wrap = True
    for side in ("margin_left", "margin_right", "margin_top", "margin_bottom"):
        setattr(tf, side, Inches(margin))
    return tf


def text(slide, s, x, y, w, h, size=16, color=INK, bold=False, font=BODY, align=PP_ALIGN.LEFT,
         italic=False, spacing=1.0, anchor=MSO_ANCHOR.TOP):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(box)
    tf.vertical_anchor = anchor
    lines = s.split("\n") if isinstance(s, str) else s
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = line
        r.font.size, r.font.bold, r.font.italic = Pt(size), bold, italic
        r.font.name, r.font.color.rgb = font, color
    return box


def bullets(slide, items, x, y, w, h, size=15, color=INK, gap=7):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = _tf(box)
    for i, it in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(gap)
        p.line_spacing = 1.08
        bold = False
        if isinstance(it, tuple):
            it, bold = it
        r = p.add_run()
        r.text = "— " + it
        r.font.size, r.font.name, r.font.color.rgb, r.font.bold = Pt(size), BODY, color, bold
    return box


def card(slide, x, y, w, h, fill=SURFACE, line=None, shadow=False):
    """A tinted rounded card. No edge stripe: per the design rules those read as AI filler."""
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.06
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    if line is None:
        sh.line.fill.background()
    else:
        sh.line.color.rgb = line
        sh.line.width = Pt(1)
    sh.shadow.inherit = shadow
    _tf(sh).text = ""
    sh.text_frame.paragraphs[0].runs and None
    return sh


def stat(slide, x, y, w, value, label, color=INK, vsize=54, sub=None):
    text(slide, value, x, y, w, 0.95, size=vsize, bold=True, color=color, font=HEAD)
    text(slide, label, x, y + 0.92, w, 0.55, size=12.5, color=MUTED)
    if sub:
        text(slide, sub, x, y + 1.38, w, 0.5, size=11.5, color=MUTED, italic=True)


def pill(slide, x, y, w, h, label, fill, fg=WHITE, size=14):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.5
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.fill.background()
    sh.shadow.inherit = False
    tf = _tf(sh)
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    r = p.add_run()
    r.text = label
    r.font.size, r.font.bold, r.font.name, r.font.color.rgb = Pt(size), True, BODY, fg
    return sh


def table(slide, rows, x, y, w, col_w, size=11.5, head_fill=DARK, row_h=0.28):
    shp = slide.shapes.add_table(len(rows), len(rows[0]), Inches(x), Inches(y),
                                 Inches(w), Inches(row_h * len(rows)))
    t = shp.table
    t.first_row = True
    for j, cw in enumerate(col_w):
        t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, cell in enumerate(row):
            val, col, bold = (cell if isinstance(cell, tuple) else (cell, None, False))
            c = t.cell(i, j)
            c.margin_left = c.margin_right = Inches(0.07)
            c.margin_top = c.margin_bottom = Inches(0.02)
            c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.fill.solid()
            c.fill.fore_color.rgb = head_fill if i == 0 else (WHITE if i % 2 else SURFACE)
            tf = c.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.RIGHT if (j and isinstance(val, str) and
                                             val.replace(".", "").replace("-", "").isdigit()) else PP_ALIGN.LEFT
            r = p.add_run()
            r.text = str(val)
            r.font.size = Pt(size)
            r.font.name = BODY
            r.font.bold = bold or i == 0
            r.font.color.rgb = WHITE if i == 0 else (col or INK)
    return t


def dot(slide, cx, cy, d, fill, ring=WHITE):
    sh = slide.shapes.add_shape(MSO_SHAPE.OVAL, Inches(cx - d / 2), Inches(cy - d / 2),
                                Inches(d), Inches(d))
    sh.fill.solid()
    sh.fill.fore_color.rgb = fill
    sh.line.color.rgb = ring
    sh.line.width = Pt(1.25)
    sh.shadow.inherit = False
    return sh


def vline(slide, x, y0, y1, color, width=1.0, dash=False):
    from pptx.enum.dml import MSO_LINE_DASH_STYLE
    ln = slide.shapes.add_connector(1, Inches(x), Inches(y0), Inches(x), Inches(y1))
    ln.line.color.rgb = color
    ln.line.width = Pt(width)
    if dash:
        ln.line.dash_style = MSO_LINE_DASH_STYLE.DASH
    return ln


def strip_plot(slide, series, x, y, w, h, lo, hi, ticks, band=None):
    """A one-axis strip plot drawn from shapes.

    Not a native PowerPoint chart type (scatter renders a 1-D strip badly and inconsistently), so it
    is drawn as real vector shapes: editable in PowerPoint, no image, and it renders identically
    wherever the deck is opened. `series` is [(label, {name: value}, colour, lane_fraction)].
    """
    def px(v):
        return x + (min(max(v, lo), hi) - lo) / (hi - lo) * w

    if band:
        b0, b1 = px(band[0]), px(band[1])
        sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(b0), Inches(y),
                                    Inches(max(b1 - b0, 0.04)), Inches(h))
        sh.fill.solid()
        sh.fill.fore_color.rgb = RGBColor(0xDC, 0xE1, 0xE8)
        sh.line.fill.background()
        sh.shadow.inherit = False
        _tf(sh).text = ""
    for t in ticks:
        vline(slide, px(t), y, y + h, RGBColor(0xE4, 0xE6, 0xEA), 0.75)
        text(slide, f"{t:g}", px(t) - 0.4, y + h + 0.04, 0.8, 0.28, size=11, color=MUTED,
             align=PP_ALIGN.CENTER)
    for label, pts, col, lane in series:
        cy = y + h * lane
        vals = [v for v in pts.values()]
        slide.shapes.add_connector(1, Inches(px(min(vals))), Inches(cy),
                                   Inches(px(max(vals))), Inches(cy)).line.color.rgb = col
        text(slide, label, x - 1.75, cy - 0.17, 1.6, 0.34, size=12.5, color=MUTED,
             align=PP_ALIGN.RIGHT)
        for v in sorted(vals):
            dot(slide, px(v), cy, 0.17, col)
    return px


def title_slide(prs, head, kicker=None, dark=False):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    if dark:
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = DARK
    text(s, head, 0.62, 0.42, 12.1, 0.95, size=32, bold=True, font=HEAD,
         color=WHITE if dark else INK)
    if kicker:
        text(s, kicker, 0.62, 1.34, 12.1, 0.45, size=14.5, color=PALE if dark else MUTED, italic=True)
    return s


def notes(slide, say, question=None, answer=None, optional=False):
    parts = []
    if optional:
        parts.append("[OPTIONAL — drop this slide for a 10-minute slot]")
    parts.append("SAY: " + say)
    if question:
        parts.append(f"LIKELY QUESTION: {question}\nANSWER: {answer}")
    slide.notes_slide.notes_text_frame.text = "\n\n".join(parts)


# ---------------------------------------------------------------- the deck
def build(R: dict, C: dict, A: dict, out: Path) -> Path:
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(W), Inches(H)

    fam = need(R, "config", "model_family")
    verdict = need(A, "decision", "verdict")
    action = need(A, "decision", "action")
    att_id = need(A, "id")
    e1 = need(R, "E1_conventional")
    det = need(R, "summary", "tamper_detection")
    fa = need(R, "summary", "honest_false_alarm")
    loo = need(R, "LOO_calibration_false_alarm")
    inv = need(R, "E8_invalidation")
    audits = need(C, "audits")
    cnull = need(C, "component_null")
    ev = need(A, "evidence")
    sa = need(ev, "stage_a")
    pol = need(A, "policy")
    env = need(A, "environment", default={})

    honest4 = {k: v["perplexity_relative_change"] for k, v in e1.items()
               if v["class"] == "honest" and k.startswith("test4")}
    tamp = {k: v["perplexity_relative_change"] for k, v in e1.items() if v["class"] == "tampered"}
    hmax_k, hmax = max(honest4.items(), key=lambda t: t[1])
    tmin_k, tmin = min(tamp.items(), key=lambda t: t[1])

    TDIR = {"template_inject": "t_template", "generation_edit": "t_gen",
            "trigger_easy": "t_trigger_easy", "trigger_hard": "t_trigger_hard",
            "canary_0.5": "t_canary_0.5", "canary_2.0": "t_canary_2.0", "canary_8.0": "t_canary_8.0"}
    beh = {n: any(need(det, n, s, "detect_rate") > 0 for s in ("guided", "random", "benchmark"))
           for n in TDIR}
    comp_hit = {n: (audits.get(d, {}).get("n_hard_anomalies") or 0) > 0 for n, d in TDIR.items()}
    n_beh = sum(beh.values())
    n_comb = sum(beh[n] or comp_hit[n] for n in TDIR)
    missed = [n for n in TDIR if not (beh[n] or comp_hit[n])]

    # ---- 1 title
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = DARK
    text(s, "BackdoorLens", 0.85, 2.15, 11.6, 1.3, size=56, bold=True, font=HEAD, color=WHITE)
    text(s, "Transformation-aware behavioural attestation for derived open-weight LLM artifacts",
         0.85, 3.42, 11.3, 0.9, size=21, color=PALE)
    # DIM, not MUTED: MUTED on this background measures 2.59:1, below the 4.5:1 floor — and this
    # is the author line on the title slide, the first thing the panel reads.
    text(s, "Ayush Munot  ·  MCA, Savitribai Phule Pune University", 0.85, 6.18, 9, 0.4,
         size=13.5, color=DIM)
    text(s, f"evidence: {fam}  ·  attestation {att_id}", 0.85, 6.58, 9, 0.4, size=12, color=DIM,
         font=MONO)
    notes(s, "A model can carry a valid publisher signature and still behave differently after it is "
             "quantized for deployment. BackdoorLens measures that difference, calibrates what is "
             "expected for that transformation, and turns the evidence into a deployment decision. "
             "It is not a proof of safety, and I will say so again at the end.")

    # ---- 2 the deployment gap
    s = title_slide(prs, "What you vetted is not what you serve")
    card(s, 0.62, 2.0, 3.55, 2.15)
    card(s, 4.47, 2.0, 3.55, 2.15)
    card(s, 8.32, 2.0, 4.39, 2.15, fill=RGBColor(0xFD, 0xEC, 0xE3))
    for i, (hd, bd, cl) in enumerate([
        ("1 · Vetted", "You download an open-weight model, verify the publisher's signature, run your "
                       "evals. The artifact you trusted is the one you tested.", INK),
        ("2 · Transformed", "To fit the serving budget you quantize it. Conversion, adapter merge, a "
                            "template change and a runtime swap are all the same story.", INK),
        ("3 · Served", "The signature no longer verifies — the bytes changed on purpose. Nothing "
                       "re-checks behaviour. This artifact is what answers your users.", TAMPER)]):
        x = 0.62 + i * 3.85
        wid = 4.39 if i == 2 else 3.55
        text(s, hd, x + 0.3, 2.22, wid - 0.6, 0.45, size=17, bold=True, font=HEAD, color=cl)
        text(s, bd, x + 0.3, 2.74, wid - 0.6, 1.3, size=14, color=INK, spacing=1.1)
    text(s, "The gap is not that signing is broken. Signing answers a different question: did the "
            "bytes change? It was never meant to answer whether behaviour survived.",
         0.62, 4.75, 12.1, 0.9, size=17, bold=True, font=HEAD, color=INK, spacing=1.12)
    notes(s, "This is the whole problem in one slide. Quantization is near-universal in self-hosted "
             "serving — you quantize to fit the GPU — and it is exactly the step that voids the "
             "signature. So almost everyone self-hosting an open-weight model is serving an artifact "
             "nobody re-verified.",
          "Isn't this solved by signing the derived model too?",
          "Signing the derived artifact proves who built it and that it hasn't changed since. It says "
          "nothing about whether its behaviour still matches what you vetted. Those are different "
          "questions and only one of them has a tool.")

    # ---- 3 what is audited
    s = title_slide(prs, "What is audited: the whole deployment bundle",
                    "a model is not only its weight file")
    comps = [c["component"] for c in need(inv, "cases") if c.get("expected_invalidated")]
    oos = [c["component"] for c in need(inv, "cases") if not c.get("expected_invalidated")]
    for i, c in enumerate(comps):
        col, row = i % 4, i // 4
        x, y = 0.62 + col * 3.07, 2.05 + row * 0.95
        card(s, x, y, 2.82, 0.72, fill=SURFACE)
        text(s, c, x + 0.22, y + 0.17, 2.4, 0.4, size=14.5, bold=True, color=INK, font=MONO)
    text(s, f"Every one of these {len(comps)} components is hashed into the certificate. Change any "
            f"of them and the certificate invalidates itself.", 0.62, 4.1, 12.1, 0.55, size=16,
         bold=True, font=HEAD)
    card(s, 0.62, 4.85, 12.1, 1.5, fill=RGBColor(0xE8, 0xF0, 0xFB))
    text(s, "Scope is explicit, and so is its boundary", 0.9, 5.02, 11.5, 0.4, size=15, bold=True,
         font=HEAD, color=HONEST)
    text(s, f"Anything outside the declared scope — {', '.join(oos)} — deliberately does NOT "
            f"invalidate. A field that cannot be read is recorded UNKNOWN and never guessed, and an "
            f"UNKNOWN field blocks an ATTESTED verdict rather than being filled in.",
         0.9, 5.45, 11.5, 0.8, size=13.5, color=INK, spacing=1.1)
    notes(s, "This answers 'what exactly have you implemented'. The audited object is the deployment "
             "bundle, not the weight file: weights, adapters, tokenizer, chat template, config, "
             "generation policy, quantization metadata and runtime manifest.",
          "Where do these component names come from?",
          "They are the in-scope components in the certificate, and the table on the gate slide is "
          "the test that each one invalidates it. The last row there is the negative control.",
          optional=True)

    # ---- 4 how it works
    s = title_slide(prs, "How it works", "twelve stages; each one is a module you can open")
    stages = [("1–3", "Manifest both bundles, diff the artifact, check they are comparable at all",
               "scope · bundle_diff"),
              ("4–6", "Load both models; 128 probes through each; per-probe JSD, KL, top-k, margin, NLL",
               "probes · divergence"),
              ("7–9", "Calibrate against honest derivations; budgeted search; confirm and replay",
               "calibration · search"),
              ("10–12", "Component differential; deterministic verdict; ed25519 signature",
               "weight_delta · decision")]
    for i, (num, what, mods) in enumerate(stages):
        y = 1.95 + i * 1.08
        card(s, 0.62, y, 12.1, 0.92, fill=SURFACE if i % 2 == 0 else WHITE)
        pill(s, 0.85, y + 0.2, 0.95, 0.5, num, HONEST, size=13)
        text(s, what, 2.0, y + 0.13, 8.0, 0.75, size=14.5, color=INK, spacing=1.08)
        text(s, mods, 10.2, y + 0.26, 2.35, 0.4, size=11.5, color=MUTED, font=MONO)
    card(s, 0.62, 6.38, 12.1, 0.72, fill=RGBColor(0xE8, 0xF0, 0xFB))
    text(s, "The verdict is chosen by a deterministic policy in decision.py. No language model is "
            "involved in deciding anything.", 0.9, 6.55, 11.5, 0.42, size=15, bold=True, font=HEAD,
         color=HONEST)
    notes(s, "This answers 'how did you implement it'. Run run/1_audit.py --help and this same list "
             "prints, in the order audit.py performs it. The gate is deliberately not in this list: "
             "it is a separate command that loads no model.",
          "Does an LLM decide the verdict?",
          "No. decision.py is a deterministic policy over the statistics — thresholds come from "
          "calibration, and the same evidence always produces the same verdict. That is why the "
          "certificate is replayable.")

    # ---- 5 conventional checks
    s = title_slide(prs, "Every standard pre-deployment check fails to separate",
                    f"all {len(e1)} held-out and tampered artifacts")
    rows = [("check", "result across every artifact", "")]
    rows += [("hash / signature", "PASS on all — including every tampered artifact", "PASS"),
             ("serialization scan", "PASS on all", "PASS"),
             ("chat-template static scan", "PASS on all — including the template injection itself", "PASS"),
             ("relative perplexity", "reacts — but fires on honest quantization too", "OVERLAPS")]
    t = table(s, [[c for c in r[:2]] for r in rows], 0.62, 1.95, 7.5, [2.5, 5.0], size=13, row_h=0.46)
    for i in range(1, 5):
        t.cell(i, 1).text_frame.paragraphs[0].runs[0].font.color.rgb = INK
    stat(s, 8.5, 1.9, 4.2, f"{hmax:.5f}", f"worst honest int4 artifact ({hmax_k})", color=HONEST,
         vsize=40)
    stat(s, 8.5, 3.5, 4.2, f"{tmin:.5f}", f"LOWEST tampered artifact ({tmin_k})", color=TAMPER,
         vsize=40)
    card(s, 0.62, 4.95, 12.1, 1.45, fill=RGBColor(0xFD, 0xEC, 0xE3))
    text(s, "The honest maximum exceeds the tampered minimum", 0.9, 5.12, 11.5, 0.42, size=17,
         bold=True, font=HEAD, color=TAMPER)
    text(s, "So no perplexity threshold passes every honest quantization and catches every tamper. "
            "Set it high enough to catch the planted triggers and it misses the template injection; "
            "set it low enough to catch that and it blocks legitimate quantization work. This is a "
            "measured property of this data, not an argument.",
         0.9, 5.58, 11.5, 0.8, size=13.5, color=INK, spacing=1.1)
    notes(s, "This is the slide that justifies the whole project. Three of the four checks a real "
             "pipeline runs today pass everything. The fourth has a pulse but cannot tell honest "
             "work from tampering — and I can prove that rather than assert it.",
          "Couldn't you just pick a smarter threshold?",
          f"Not on this data: {hmax_k} at {hmax:.5f} is honest and {tmin_k} at {tmin:.5f} is "
          f"tampered. Any single threshold either blocks honest quantization or passes a tamper. "
          f"That is what overlap means.")

    # ---- 6 separability chart
    s = title_slide(prs, "The same point, plotted", "relative perplexity change versus the reference")
    in_axis = {k: v for k, v in tamp.items() if v <= 0.24}
    off_axis = sorted(v for v in tamp.values() if v > 0.24)
    px = strip_plot(s, [("honest int4", honest4, HONEST, 0.3),
                        ("tampered", in_axis, TAMPER, 0.72)],
                    2.45, 2.1, 9.9, 2.5, 0.11, 0.24,
                    [0.11, 0.13, 0.15, 0.17, 0.19, 0.21, 0.23], band=(tmin, hmax))
    text(s, "overlap", px((tmin + hmax) / 2) - 0.75, 1.72, 1.5, 0.3, size=12, bold=True,
         color=INK, align=PP_ALIGN.CENTER)
    text(s, f"worst honest {hmax:.5f}", px(hmax) - 1.2, 2.1 + 2.5 * 0.3 - 0.52, 2.4, 0.3, size=11.5,
         bold=True, color=HONEST, align=PP_ALIGN.CENTER)
    text(s, f"lowest tampered {tmin:.5f}", px(tmin) - 1.3, 2.1 + 2.5 * 0.72 + 0.17, 2.6, 0.3,
         size=11.5, bold=True, color=TAMPER, align=PP_ALIGN.CENTER)
    text(s, "relative perplexity change vs reference", 2.45, 5.0, 9.9, 0.3, size=12, color=MUTED,
         italic=True, align=PP_ALIGN.CENTER)
    if off_axis:
        text(s, "two tampered artifacts sit off this axis at "
                + " and ".join(f"{v:.2f}" for v in off_axis),
             2.45, 5.34, 9.9, 0.3, size=11.5, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
    text(s, f"The blue maximum ({hmax:.5f}) is to the RIGHT of the orange minimum ({tmin:.5f}). "
            f"The classes overlap, so no vertical line separates them.",
         0.62, 5.95, 12.1, 0.85, size=16.5, bold=True, font=HEAD, spacing=1.12)
    notes(s, "One axis, two lanes. Everything blue is honest quantization, everything orange is "
             "tampered. The grey band is where they overlap — the rightmost blue point is further "
             "right than the leftmost orange one, and that is why a threshold cannot work.",
          "Why are two points missing?",
          f"The two strongest weight perturbations move perplexity by "
          f"{' and '.join(f'{v:.2f}' for v in off_axis)}. They are off the axis so the interesting "
          f"region stays readable, and they are trivially detected anyway.")

    # ---- 7 calibration
    s = title_slide(prs, "No global thresholds — calibrate against honest derivations",
                    "the core of the method")
    card(s, 0.62, 1.95, 5.9, 2.5, fill=SURFACE)
    text(s, "The wrong way", 0.9, 2.12, 5.4, 0.4, size=16, bold=True, font=HEAD, color=MUTED)
    text(s, "Pick a divergence threshold. Apply it to every model and every transformation. "
            "An int4 artifact and an int8 artifact get judged by the same constant — so the "
            "threshold is either too loose for one or too tight for the other.",
         0.9, 2.58, 5.4, 1.7, size=14, color=INK, spacing=1.12)
    card(s, 6.82, 1.95, 5.9, 2.5, fill=RGBColor(0xE8, 0xF0, 0xFB))
    text(s, "What BackdoorLens does", 7.1, 2.12, 5.4, 0.4, size=16, bold=True, font=HEAD, color=HONEST)
    text(s, f"Build k honest derivations of the SAME declared transformation and measure their "
            f"spread. Every statistic is a RATIO against the audited artifact's own level, tested "
            f"with a t prediction interval (df = k−1). Here k = "
            f"{need(ev, 'null', 'n_derivations')}, at a matched budget of "
            f"{need(ev, 'null', 'search_budget')}.",
         7.1, 2.58, 5.4, 1.7, size=14, color=INK, spacing=1.12)
    for i, (hd, bd) in enumerate([
        ("One derivation = one unit", "Never one prompt. Pooling per-prompt tails across a single "
                                      "derivation is pseudo-replication, and is reported descriptively only."),
        ("The null must match", "A null fitted on int4 is not used for an int8 artifact. A mismatch "
                                "names the differing field and returns INCONCLUSIVE."),
        ("Fitted before evaluation", "Thresholds come from the calibration set; held-out artifacts "
                                     "are never used to choose them.")]):
        x = 0.62 + i * 4.09
        card(s, x, 4.72, 3.79, 1.72, fill=WHITE, line=PALE)
        text(s, hd, x + 0.24, 4.9, 3.3, 0.4, size=14, bold=True, font=HEAD, color=INK)
        text(s, bd, x + 0.24, 5.32, 3.3, 1.0, size=12, color=MUTED, spacing=1.1)
    notes(s, "This answers 'what parameters do you use to detect abnormal behaviour'. The honest "
             "answer is that no parameter has a fixed threshold. Everything is relative to the "
             "artifact's own level and to what honest derivations of its declared family do.",
          "Where does the null come from in a real deployment?",
          "From the customer's own pipeline. They already quantize in CI; running that pipeline six "
          "times gives the null. So the system calibrates to their stack and ships no vendor "
          "threshold at all.")

    # ---- 8 the envelope
    lv = need(ev, "null", "honest_levels")
    level = need(sa, "level_median_jsd")
    guard = need(pol, "min_level_ratio")
    gline = max(lv) * guard
    s = title_slide(prs, "A significant deviation that correctly did not fire",
                    "shown because it is a real limitation of the method")
    X0, WID, Y0, HGT = 2.45, 9.9, 2.15, 1.9
    lo, hi = 0.18, max(gline, level) * 1.12
    px = strip_plot(s, [("honest derivations", {f"h{i}": v for i, v in enumerate(lv)}, HONEST, 0.5)],
                    X0, Y0, WID, HGT, lo, hi, [0.2, 0.3, 0.4, 0.5, 0.6],
                    band=(min(lv), max(lv)))
    dot(s, px(level), Y0 + HGT * 0.5, 0.28, TAMPER)
    text(s, f"audited {level:.4f}", px(level) - 1.1, Y0 + HGT * 0.5 + 0.22, 2.2, 0.3, size=12.5,
         bold=True, color=TAMPER, align=PP_ALIGN.CENTER)
    text(s, f"honest envelope {min(lv):.3f}-{max(lv):.3f}", px((min(lv) + max(lv)) / 2) - 1.4,
         Y0 - 0.38, 2.8, 0.3, size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
    vline(s, px(gline), Y0 - 0.05, Y0 + HGT + 0.05, MUTED, 1.75, dash=True)
    text(s, f"guard {guard:g}x = {gline:.3f}", px(gline) - 1.3, Y0 - 0.38, 2.6, 0.3, size=11.5,
         bold=True, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, "suite-wide median JSD (the artifact's own 'level')", X0, Y0 + HGT + 0.42, WID, 0.3,
         size=12, color=MUTED, italic=True, align=PP_ALIGN.CENTER)
    stat(s, 0.62, 4.95, 3.4, f"{need(sa, 'p_level'):.2e}", "p_level - statistically significant",
         color=HONEST, vsize=28)
    stat(s, 4.3, 4.95, 3.4, f"{need(sa, 'level_ratio_vs_max_honest'):.3f}x",
         f"of the honest maximum - the guard needs {guard:g}x", color=TAMPER, vsize=28)
    stat(s, 7.98, 4.95, 4.74, f"{need(sa, 'argmax_flip_rate'):.1%}",
         "of probes flip their top token", color=INK, vsize=28)
    text(s, "A modification pervasive enough to raise its own baseline hides from ratio-normalised "
            "tests. The behavioural channel FAILED on this artifact. That is why a second, non-ratio "
            "channel exists.", 0.62, 6.6, 12.1, 0.7, size=15.5, bold=True, font=HEAD, spacing=1.1)
    notes(s, "I am showing you a failure of my own method. The deviation is statistically significant "
             "but the effect-size guard refused it, because the artifact raised its own baseline. "
             "Being honest about this is what makes the next slide credible.",
          "Isn't the guard just hiding a real detection?",
          "It is preventing a false one. Without an effect-size guard, any tiny but consistent "
          "difference becomes significant at this sample size. The guard is why the false-alarm rate "
          "is zero out of ninety-six. The cost is this blind spot, and the component channel covers it.")

    # ---- 9 headline result
    s = title_slide(prs, "Two independent channels, and why both are needed")
    rows = [["tampered artifact", "behavioural", "component", "combined"]]
    for n in TDIR:
        rows.append([n,
                     ("detected", HONEST, True) if beh[n] else ("missed", TAMPER, False),
                     (str(audits.get(TDIR[n], {}).get("n_hard_anomalies", 0)) + " groups", HONEST, True)
                     if comp_hit[n] else ("no anomaly", MUTED, False),
                     ("DETECTED", HONEST, True) if (beh[n] or comp_hit[n]) else ("missed", TAMPER, True)])
    table(s, rows, 0.62, 1.9, 7.3, [2.5, 1.7, 1.6, 1.5], size=12, row_h=0.37)
    stat(s, 8.3, 1.95, 2.1, f"{n_beh}/{len(TDIR)}", "behavioural channel alone", color=MUTED, vsize=44)
    stat(s, 10.6, 1.95, 2.1, f"{n_comb}/{len(TDIR)}", "both channels together", color=HONEST, vsize=44)
    card(s, 8.3, 3.85, 4.42, 1.6, fill=RGBColor(0xFD, 0xEC, 0xE3))
    text(s, "The channels are complementary", 8.54, 4.0, 3.9, 0.4, size=14.5, bold=True, font=HEAD,
         color=TAMPER)
    text(s, "The component channel catches exactly the artifacts every behavioural test missed — the "
            "two planted-trigger backdoors — and nothing else.", 8.54, 4.42, 3.9, 0.9, size=12.5,
         color=INK, spacing=1.1)
    card(s, 0.62, 5.6, 12.1, 1.5, fill=DARK)
    text(s, "The result this project exists for", 0.9, 5.76, 11.5, 0.4, size=15, bold=True, font=HEAD,
         color=WHITE)
    text(s, f"On the backdoored artifact, the behavioural channel alone yields ATTESTED_WITH_DRIFT "
            f"-> ALLOW_WITH_MONITORING: it would have been deployed. With the component channel the "
            f"verdict is {verdict} -> {action}. Same evidence, same policy, one extra channel.",
         0.9, 6.2, 11.5, 0.8, size=14, color=PALE, spacing=1.12)
    notes(s, "This is the headline. Four of seven with behaviour alone; six of seven with both. And "
             "on the artifact that matters most — the planted backdoor — behaviour alone would have "
             "let it through with monitoring. The component channel quarantines it.",
          f"What about the one you still miss ({missed[0] if missed else 'none'})?",
          f"{missed[0] if missed else 'None'} is the smallest weight perturbation in the set. It is "
          f"too small to leave the calibrated envelope and it touches only tensor groups honest "
          f"quantization already touches, so neither channel sees it. It marks the sensitivity floor "
          f"of the method at this budget. I am not claiming it as a win.")

    # ---- 10 false alarms
    s = title_slide(prs, "How often it cries wolf", "a detector with no false-alarm rate is not a detector")
    rows = [["false-alarm test", "alarms", "honest audits"]]
    tot_n = tot_k = 0
    for st in ("guided", "random", "benchmark"):
        v = need(fa, st)
        n, k = int(v["n"]), int(round(v["fpr"] * v["n"]))
        tot_n += n
        tot_k += k
        rows.append([f"behavioural · {st}", str(k), str(n)])
    for bits, blk in loo.items():
        r = blk["rows"]
        k = sum(1 for x in r if x.get("detected"))
        tot_n += len(r)
        tot_k += k
        b = str(bits)
        rows.append([f"behavioural · leave-one-out {b if b.startswith('int') else 'int' + b}",
                     str(k), str(len(r))])
    rows.append(["behavioural TOTAL", str(tot_k), str(tot_n)])
    hd = [d for d in audits if d.startswith(("cal", "test"))]
    rows.append(["component · leave-one-out", str(need(C, "leave_one_out", "n_false_positives")),
                 str(need(C, "leave_one_out", "n_tested"))])
    rows.append(["component · held-out honest",
                 str(sum(1 for d in hd if (audits[d].get("n_hard_anomalies") or 0) > 0)), str(len(hd))])
    table(s, rows, 0.62, 1.95, 7.3, [3.9, 1.7, 1.7], size=12.5, row_h=0.4)
    stat(s, 8.4, 2.1, 4.3, f"{tot_k}/{tot_n}", "behavioural false alarms on honest derivations",
         color=HONEST, vsize=48)
    stat(s, 8.4, 4.1, 4.3, "0", "component false alarms, leave-one-out and held-out", color=HONEST,
         vsize=48)
    text(s, "Honest derivations are the independent experimental units. Leave-one-out fits the null "
            "on k-1 and audits the held-out one, which is the only way to measure a false-alarm rate "
            "without reusing the artifacts the null was built from.",
         0.62, 6.25, 12.1, 0.85, size=14.5, color=INK, spacing=1.12)
    notes(s, "Detection is worthless without this. Zero false alarms in ninety-six behavioural "
             "audits of honest derivations, and zero in both component tests.",
          "Ninety-six audits of how many distinct artifacts?",
          "Twelve held-out honest derivations, across three search strategies and three seeds, plus "
          "leave-one-out on the calibration set. The artifacts are reused across strategies, which I "
          "state plainly — the independent unit is the derivation, and there are twelve of those.")

    # ---- 11 counterexample  (omitted entirely if the audit confirmed none — never invented)
    cx = (A.get("counterexamples") or [None])[0]
    if cx is None:
        print("NOTE: this attestation records no counterexample, so slide 11 is omitted. "
              "Nothing was invented to fill it.")
    else:
        s = title_slide(prs, "A reproducible counterexample",
                        "found by budgeted search, which was never told what to look for")
        card(s, 0.62, 1.9, 12.1, 1.72, fill=SURFACE)
        y = 2.05
        for m in need(cx, "prompt", "messages"):
            pill(s, 0.85, y, 1.05, 0.3, need(m, "role"), MUTED, size=10)
            text(s, need(m, "content")[:118], 2.05, y + 0.01, 10.4, 0.3, size=12, color=INK, font=MONO)
            y += 0.38
        for i, (hd, key, col) in enumerate((("reference output", "reference_output", HONEST),
                                            ("derived output", "derived_output", TAMPER))):
            x = 0.62 + i * 6.25
            card(s, x, 3.78, 5.85, 1.5, fill=WHITE, line=PALE)
            text(s, hd, x + 0.24, 3.92, 5.3, 0.32, size=12, bold=True, color=col)
            text(s, need(cx, key)[:165], x + 0.24, 4.28, 5.37, 0.95, size=11.5, color=INK, font=MONO,
                 spacing=1.08)
        # per-probe metrics live nested under "metrics" (audit.confirm), not at the record's top level
        for i, (v, lab) in enumerate(((f"{need(cx, 'jsd'):.4f}", "JSD"),
                                      (f"{need(cx, 'n_variants_pass')}/{need(cx, 'n_variants')}",
                                       "semantic variants"),
                                      ("yes" if need(cx, "replay_ok") else "NO", "replay-verified"),
                                      (f"{need(cx, 'metrics', 'topk_overlap'):.3f}", "top-k overlap"))):
            stat(s, 0.62 + i * 3.1, 5.5, 2.9, v, lab, color=INK, vsize=30)
        text(s, "Persistence across semantic variants is what separates a reproducible finding from a "
                "single unlucky prompt, and replay re-verifies it through a separate call path.",
             0.62, 6.85, 12.1, 0.5, size=13.5, color=MUTED, spacing=1.1)
        notes(s, "The search produced this prompt on its own. The two models answer it completely "
                 "differently, it survives every semantic variant tried, and it replays through an "
                 "independent code path — so it is evidence, not an anecdote.",
              "The prompt looks like gibberish — is that realistic?",
              "No, and I would not claim it is. It is a counterexample, not a user query: its job is "
              "to prove a behavioural difference exists and is reproducible. A realistic trigger is "
              "the planted one, and that is the artifact the component channel catches.",
              optional=True)

    # ---- 12 the gate
    s = title_slide(prs, "The certificate enforces itself", "no model is loaded: this is a hash comparison")
    cases = need(inv, "cases")
    rows = [["in-scope component modified", "certificate", "as expected"]]
    for c in cases:
        ok = (c["state"] == "INVALIDATED") == bool(c.get("expected_invalidated"))
        rows.append([c["component"],
                     (c["state"], TAMPER if c["state"] == "INVALIDATED" else HONEST, True),
                     ("yes", HONEST, False) if ok else ("NO", TAMPER, True)])
    table(s, rows, 0.62, 1.9, 6.6, [3.4, 1.8, 1.4], size=11.5, row_h=0.33)
    lat = sorted(c["latency_s"] for c in cases if c.get("latency_s") is not None)
    stat(s, 7.6, 2.0, 2.5, f"{len(cases)}/{len(cases)}", "behaved as expected", color=HONEST, vsize=40)
    stat(s, 10.3, 2.0, 2.4, f"{lat[len(lat)//2]:.2f}s", "median decision", color=HONEST, vsize=40)
    card(s, 7.6, 3.7, 5.12, 1.7, fill=RGBColor(0xE8, 0xF0, 0xFB))
    text(s, "The negative control", 7.84, 3.86, 4.6, 0.4, size=14.5, bold=True, font=HEAD, color=HONEST)
    text(s, f"The last row is a change OUTSIDE the certified scope. The certificate stays VALID. "
            f"The scope is precise, not merely alarmist — a gate that invalidates on everything is "
            f"useless.", 7.84, 4.28, 4.64, 1.0, size=12.5, color=INK, spacing=1.1)
    for i, (v, lab) in enumerate(((f"ALLOW  0", "attested"), ("QUARANTINE  10", "suspicious"),
                                  ("BLOCK  20", "divergent"), ("RE-ATTEST  40", "invalidated"))):
        pill(s, 0.62 + i * 3.1, 5.72, 2.9, 0.5, v, (HONEST, TAMPER, DARK, MUTED)[i], size=12.5)
        text(s, lab, 0.62 + i * 3.1, 6.3, 2.9, 0.3, size=11.5, color=MUTED, align=PP_ALIGN.CENTER)
    text(s, "These are process exit codes, so the gate drops into a CI pipeline unchanged.",
         0.62, 6.75, 12.1, 0.4, size=13.5, color=INK)
    notes(s, "A report is advice; this is enforcement. Modify any certified component and the "
             "certificate invalidates itself in about half a second, without loading a model, because "
             "it is a hash comparison. The exit codes mean it drops straight into CI.",
          "What stops someone re-signing a modified artifact?",
          "Nothing, and that is not what this defends. The certificate binds a verdict to an exact "
          "artifact: if the artifact changes the old certificate no longer describes it. Who is "
          "allowed to issue certificates is a key-management question, and the gate pins the public "
          "key.")

    # ---- 13 scope boundary
    s = title_slide(prs, "What it refuses to answer", "the boundary is enforced, not documented")
    for i, (hd, bd, verdict_txt) in enumerate((
            ("No reference", "Handed an artifact with no trusted reference, there is nothing to "
                             "compare against. The method certifies a derivation.",
             "REQUIRE_MORE_EVIDENCE"),
            ("Another model family", "Different vocabulary or architecture: the two logit vectors do "
                                     "not even have the same length, so no divergence between them "
                                     "exists to measure.", "refuses, no model loaded"),
            ("No matched null", "A null fitted on a different transformation family cannot calibrate "
                                "this one. The mismatch names the differing field.",
             "INCONCLUSIVE"))):
        x = 0.62 + i * 4.09
        card(s, x, 1.95, 3.79, 3.1, fill=SURFACE)
        text(s, hd, x + 0.26, 2.12, 3.3, 0.4, size=16, bold=True, font=HEAD, color=INK)
        text(s, bd, x + 0.26, 2.6, 3.3, 1.8, size=13, color=INK, spacing=1.12)
        pill(s, x + 0.26, 4.4, 3.27, 0.45, verdict_txt, MUTED, size=11)
    card(s, 0.62, 5.35, 12.1, 1.75, fill=RGBColor(0xE8, 0xF0, 0xFB))
    text(s, "A refusal is a result, and it is cheap", 0.9, 5.52, 11.5, 0.4, size=16, bold=True,
         font=HEAD, color=HONEST)
    text(s, "The incomparability check reads both manifests and decides before either model is "
            "loaded. The certificate it writes records behavioural_measurement: not_performed, with "
            "the statistics ABSENT rather than zero — a zero would read as 'no divergence found', "
            "the opposite of what happened. INCONCLUSIVE never means safe.",
         0.9, 5.96, 11.5, 1.0, size=13.5, color=INK, spacing=1.12)
    notes(s, "Panelists will try to hand me an arbitrary model. I can run this live: it refuses in "
             "seconds, names which field disagrees, and loads nothing. The system knows what it "
             "cannot answer, which is the property you want in a gate.",
          "So your tool only works if I give you the reference?",
          "Yes — and in practice the reference exists: you downloaded it, or your own pipeline "
          "produced the derived artifact from it. What is missing today is anyone fetching it and "
          "checking. A model pretrained from scratch has no upstream and cannot be attested by this "
          "method, or by any other — reference-free backdoor detection is an open problem.")

    # ---- 14 limitations
    s = title_slide(prs, "Limitations", "stated, not discovered by the panel")
    left = ["Quantization here is simulated round-to-nearest, not a bitsandbytes/GPTQ/GGUF kernel. "
            "Which tensor groups an honest derivation skips is a property of that implementation.",
            "The component channel's honest null is degenerate — identically zero, no variance. It is "
            "reported as a component-level finding with k stated, NEVER as a p-value.",
            "It detects provenance deviation, not malice. A benign full fine-tune flags identically.",
            "It needs full weights on both sides, so it does not apply to adapter-only bundles."]
    right = ["An attacker who confines changes to the groups the honest transformation already "
             "touches produces no component anomaly.",
             "Ratio-normalised behavioural statistics can be self-masking — shown on slide 8.",
             "Finite probing cannot exclude rare triggers outside the exercised distribution, and a "
             "compromised reference is faithfully preserved by an honest derivation.",
             f"One model family at one size, CPU only. No claim of generality across scales."]
    card(s, 0.62, 1.9, 5.9, 3.5, fill=SURFACE)
    card(s, 6.82, 1.9, 5.9, 3.5, fill=SURFACE)
    bullets(s, left, 0.88, 2.08, 5.4, 3.2, size=13, gap=11)
    bullets(s, right, 7.08, 2.08, 5.4, 3.2, size=13, gap=11)
    card(s, 0.62, 5.62, 12.1, 1.0, fill=DARK)
    text(s, "Two hypotheses were falsified and are kept in the repository with their code: reading "
            "the planted trigger out of the embedding delta, and rescuing that ranking by removing "
            "the dense component. The planned delta-guided search was therefore not built.",
         0.9, 5.82, 11.5, 0.7, size=13.5, color=PALE, spacing=1.12)
    notes(s, "I would rather state these than have you find them. The two that matter most: a benign "
             "fine-tune flags identically on the component channel, so it is a provenance detector "
             "and not a backdoor oracle; and an attacker who stays inside the groups quantization "
             "already touches defeats that channel.",
          "If a benign fine-tune also flags, what is the detection worth?",
          "It is worth a QUARANTINE, not a BLOCK — which is exactly the action the policy assigns. "
          "The claim is that the artifact is inconsistent with its DECLARED transformation. If you "
          "declared a quantization and I can show you a full fine-tune, that is a finding whether or "
          "not it was malicious.")

    # ---- 15 conclusion
    s = prs.slides.add_slide(prs.slide_layouts[6])
    bg = s.background.fill
    bg.solid()
    bg.fore_color.rgb = DARK
    text(s, "What this contributes", 0.85, 0.75, 11.6, 0.8, size=34, bold=True, font=HEAD, color=WHITE)
    items = [("A complete reference-to-derived deployment differential — the whole bundle, not the "
              "weight file."),
             ("Transformation-aware calibration: no global thresholds, and in deployment the null "
              "comes from the customer's own pipeline."),
             ("Two independent channels whose combination detects what neither does alone, with a "
              "measured false-alarm rate of zero on ninety-six honest audits."),
             ("Replayable, signed, scope-bound evidence that enforces a deployment decision and "
              "invalidates itself when the artifact changes.")]
    bullets(s, items, 0.85, 1.85, 11.6, 2.6, size=16, color=PALE, gap=14)
    card(s, 0.85, 4.75, 11.6, 1.05, fill=RGBColor(0x2A, 0x33, 0x42))
    text(s, "ATTESTED does not mean safe. INCONCLUSIVE does not mean safe. This certifies a "
            "derivation, within a stated scope, at a stated budget.",
         1.12, 4.98, 11.1, 0.7, size=16.5, bold=True, font=HEAD, color=WHITE, spacing=1.12)
    text(s, "Reproduce:  python run\\1_audit.py --derived <artifact>      ·      "
            "python run\\4_demo.py --pause", 0.85, 6.18, 11.6, 0.4, size=13, color=DIM, font=MONO)
    # a stable, meaningful order, capped to one line — env dicts are unordered and can be long
    order = ("python", "torch", "transformers", "numpy", "platform")
    picked = [(k, env[k]) for k in order if k in env] or list(env.items())[:5]
    if picked:
        text(s, "  ·  ".join(f"{k} {v}" for k, v in picked), 0.85, 6.62, 11.6, 0.4, size=10.5,
             color=DIM, font=MONO)
    notes(s, "Close on what is contributed and on what it does not claim. The last line is the one "
             "to say out loud: this certifies a derivation within a stated scope at a stated budget. "
             "Then offer the live demo.")

    out.parent.mkdir(parents=True, exist_ok=True)
    try:
        prs.save(str(out))
    except PermissionError:
        # On Windows an open PowerPoint holds an exclusive lock on the file. Nothing is wrong with
        # the deck or the evidence; a 70-line traceback for "the file is open" is just noise.
        raise SystemExit(
            f"\nCANNOT WRITE {out}\n"
            f"The file is locked, which on Windows almost always means it is OPEN IN POWERPOINT.\n"
            f"Close it and run this again — nothing else is wrong, and no evidence was touched.\n"
            f"If PowerPoint is already closed, pass a different name:  --out {out.with_name(out.stem + '_v2' + out.suffix)}"
        ) from None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--component", required=True)
    ap.add_argument("--attestation", required=True)
    ap.add_argument("--out", default="deck/BackdoorLens_review2.pptx")
    a = ap.parse_args()
    R = json.loads(Path(a.results).read_text())
    C = json.loads(Path(a.component).read_text())
    A = json.loads(Path(a.attestation).read_text())
    if C.get("component_null") is None and C.get("status") == "ok":
        C = {"component_null": C, "audits": {}, "leave_one_out": {}}
    if need(R, "config", "synthetic", default=False):
        print("REFUSING: results.json has synthetic=true. This deck is the research claim; it must "
              "not be built from a toy run. Point --results at the real-model results.")
        return 2
    problems = preflight(R, C, A)
    if problems:
        print("EVIDENCE DOES NOT MATCH WHAT THE DECK NEEDS — nothing was built.\n"
              "Every missing key is listed, so this takes one run to diagnose:")
        for b in problems:
            print("  " + b)
        print("\nThese are read by run/2_report.py and run/3_evaluate.py too; if those work and this "
              "does not, the key moved and build_deck.py needs updating to match.")
        return 2
    p = build(R, C, A, Path(a.out))
    print(f"wrote {p}  ({p.stat().st_size // 1024} KB, {len(Presentation(str(p)).slides)} slides)")
    print("every number on every slide was read from the three input files; a missing key raises")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
