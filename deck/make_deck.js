// usage: node make_deck.js <results-dir> <out.pptx>   -- all numbers come from results.json / report/macros.json
const fs = require("fs"), path = require("path");
const pptxgen = require("pptxgenjs");
const SKILL = "/mnt/skills/public/pptx/scripts/apply_theme.js";
const resDir = process.argv[2], outFile = process.argv[3] || "BackdoorLens.pptx";
const R = JSON.parse(fs.readFileSync(path.join(resDir, "results.json")));
const F = JSON.parse(fs.readFileSync(path.join(resDir, "report", "findings.json")));
const M = JSON.parse(fs.readFileSync(path.join(resDir, "report", "macros.json")));
const S = R.summary, syn = !!R.config.synthetic;
const THEME = { name: "BackdoorLens", headFontFace: "Calibri", bodyFontFace: "Calibri",
  colors: { dk1: "1B2430", lt1: "FFFFFF", dk2: "0B3B6F", lt2: "EEF3F9", accent1: "1F6FEB", accent2: "2DA44E", accent3: "D29922", accent4: "CF222E", accent5: "6E7781", accent6: "8250DF", hlink: "1F6FEB", folHlink: "6E7781" } };
const pres = new pptxgen(); pres.layout = "LAYOUT_WIDE"; pres.theme = { headFontFace: "Calibri", bodyFontFace: "Calibri" };
pres.title = "BackdoorLens"; pres.author = "Ayush Munot";
const C = pres.SchemeColor, W = 13.33;
pres.defineSlideMaster({ title: "TITLE_ONLY", background: { color: "FFFFFF" }, margin: [0.5, 0.6, 0.5, 0.6],
  objects: [{ rect: { x: 0, y: 0, w: W, h: 0.12, fill: { color: "0B3B6F" } } }],
  slideNumber: { x: 12.4, y: 7.05, w: 0.6, h: 0.3, fontSize: 10, color: "6E7781" },
  });
pres.defineSlideMaster({ title: "DARK", background: { color: "0B3B6F" }, margin: 0.6,
  });
const BANNER = "SYNTHETIC VALIDATION RUN — NOT RESEARCH EVIDENCE";
function slide(title, notes) {
  const s = pres.addSlide({ masterName: "TITLE_ONLY" }); s.addText(title, { x: 0.6, y: 0.35, w: 12.1, h: 0.9, fontSize: 28, bold: true, color: "0B3B6F", valign: "top", isTextBox: true, margin: 0, fit: "shrink" }); if (notes) s.addNotes(notes); return s; }
function banner(s) { if (syn) s.addText(BANNER, { x: 8.0, y: 0.18, w: 5.0, h: 0.3, fontSize: 11, bold: true, color: "FFFFFF", fill: { color: "CF222E" }, align: "center", isTextBox: true, margin: 0 }); }
function bullets(s, items, x, y, w, h, size = 17) {
  s.addText(items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, paraSpaceAfter: 8 } })), { x, y, w, h, fontSize: size, color: "1B2430", valign: "top", isTextBox: true, margin: 0 }); }
function card(s, x, y, w, h, head, body, color = "1F6FEB") {
  s.addShape(pres.ShapeType.rect, { x, y, w, h, fill: { color: "EEF3F9" }, line: { color: "EEF3F9" }, objectName: "card" });
  s.addShape(pres.ShapeType.rect, { x, y, w: 0.09, h, fill: { color }, line: { color } });
  s.addText(head, { x: x + 0.25, y: y + 0.12, w: w - 0.4, h: 0.4, fontSize: 15, bold: true, color: "0B3B6F", isTextBox: true, margin: 0 });
  s.addText(body, { x: x + 0.25, y: y + 0.55, w: w - 0.4, h: h - 0.65, fontSize: 13, color: "1B2430", valign: "top", isTextBox: true, margin: 0 }); }
const pctf = (x) => (100 * x).toFixed(0) + "%";

// 1 title
let s = pres.addSlide({ masterName: "DARK" });
s.addText("BackdoorLens", { x: 0.8, y: 2.3, w: 11.6, h: 1.4, fontSize: 54, bold: true, color: "FFFFFF", valign: "top", isTextBox: true, margin: 0 });
s.addText("Behavioral attestation and continuous integrity enforcement for derived open-weight LLM artifacts", { x: 0.8, y: 3.9, w: 11.5, h: 0.9, fontSize: 22, color: "DCE8F7", isTextBox: true, margin: 0 });
s.addText("Ayush Munot · MCA, Savitribai Phule Pune University", { x: 0.8, y: 6.3, w: 9, h: 0.4, fontSize: 14, color: "AFC7E8", isTextBox: true, margin: 0 });
if (syn) s.addText(BANNER, { x: 0.8, y: 0.5, w: 7.5, h: 0.4, fontSize: 14, bold: true, color: "FFFFFF", fill: { color: "CF222E" }, align: "center", isTextBox: true, margin: 0 });
s.addNotes("One-line promise: a model can have a valid signature and still behave differently after quantization, conversion, packaging, template change or runtime optimization. BackdoorLens measures that difference, calibrates what is expected, keeps replayable evidence and enforces a deployment decision. It is not a proof of safety.");
// 2 deployment gap
s = slide("The deployment gap: what you vetted is not what you serve");
const steps = ["Vetted reference", "Convert / quantize", "Package template + config", "Runtime / compiler", "Served artifact"];
steps.forEach((t, i) => { const x = 0.6 + i * 2.5; s.addShape(pres.ShapeType.roundRect, { x, y: 1.9, w: 2.1, h: 1.1, fill: { color: i === 0 ? "2DA44E" : i === 4 ? "CF222E" : "EEF3F9" }, line: { color: "0B3B6F", width: 1 }, rectRadius: 0.1 });
  s.addText(t, { x, y: 1.9, w: 2.1, h: 1.1, fontSize: 15, bold: true, align: "center", valign: "middle", color: i === 0 || i === 4 ? "FFFFFF" : "0B3B6F", isTextBox: true });
  if (i < 4) s.addText("→", { x: x + 2.1, y: 2.15, w: 0.4, h: 0.6, fontSize: 24, color: "0B3B6F", align: "center", isTextBox: true, margin: 0 }); });
bullets(s, ["Each stage can change behavior; recent work shows quantization-, optimization- and template-conditioned attacks.", "Signatures prove identity, scanners check files, perplexity checks average quality — none establishes conditional behavior.", "BackdoorLens asks: is the exact deployed bundle behaviorally faithful to the vetted reference?"], 0.6, 3.5, 12.1, 3);
// 3 three checks
s = slide("Why signatures and scanners are not enough");
card(s, 0.6, 1.7, 3.9, 3.6, "Provenance", "Is this the file I expected?\nSigstore-style signing answers this. A malicious but signed derivative still verifies.", "6E7781");
card(s, 4.7, 1.7, 3.9, 3.6, "File safety", "Does it contain malicious code?\nHub scanners and template scanners answer parts of this; they do not compare behavior with a reference.", "D29922");
card(s, 8.8, 1.7, 3.9, 3.6, "Behavioral faithfulness", "Does it behave like the reference I vetted?\nThis is the question BackdoorLens measures — complementary to the other two.", "1F6FEB");
s.addText("A passing attestation is bounded evidence, not a proof of safety.", { x: 0.6, y: 5.9, w: 12.1, h: 0.5, fontSize: 18, bold: true, color: "0B3B6F", isTextBox: true, margin: 0 });
// 4 threat model
s = slide("Threat model and trust boundaries");
bullets(s, ["Reference R: conditionally trusted — must be vetted independently.", "Derived bundle D: untrusted until attested (weights, adapters, tokenizer, template, config, generation, quantization, runtime).", "Audit engine: trusted computing base. An LLM may summarise but never sets a verdict.", "Attackers: malicious publisher, compromised pipeline, template/config editor, post-certification swap, adaptive prober.", "Out of scope: black-box APIs, purification, universal safety claims. A poisoned reference stays poisoned."], 0.6, 1.6, 12.1, 5);
// 5 research question
s = slide("Research question");
s.addText("Given reference R, derived bundle D, a declared transformation family t, input distribution X and budget B — is the observed divergence abnormal relative to honest derivations of type t, persistent across valid variants and material?", { x: 0.6, y: 1.7, w: 12.1, h: 1.7, fontSize: 22, color: "1B2430", isTextBox: true, margin: 0 });
bullets(s, ["We never claim P_R = P_D for all inputs.", "“Different” is not “malicious”: honest quantization always drifts.", "Every claim is scoped to (threat model, family, distribution, budget)."], 0.6, 4.0, 12.1, 2.6);
// 6 method
s = slide("Method: five measurable components");
["Scope tuple + bundle diff", "Paired JSD measurement", "Benign null (level + shape)", "Budgeted search", "Signed attestation + gate"].forEach((t, i) => {
  const x = 0.6 + i * 2.45; card(s, x, 1.7, 2.3, 2.5, `${i + 1}`, t, ["1F6FEB", "2DA44E", "D29922", "8250DF", "CF222E"][i]); });
bullets(s, ["Deterministic policy decides — no LLM in the decision path.", "Templates are untrusted code: rendered only in a sandboxed Jinja environment.", "INCONCLUSIVE is a first-class verdict: no matching null, too few honest derivations or too little budget never auto-approves."], 0.6, 4.6, 12.1, 2.5, 16);
// 7 workflow
s = slide("Workflow");
["Register reference", "Ingest derived", "Scope manifest", "Bundle diff", "Paired probes", "Guided search", "Confirm + replay", "Verdict", "Sign + gate", "Monitor"].forEach((t, i) => {
  const x = 0.6 + (i % 5) * 2.45, y = i < 5 ? 1.9 : 3.9; s.addShape(pres.ShapeType.roundRect, { x, y, w: 2.2, h: 1.3, fill: { color: "EEF3F9" }, line: { color: "1F6FEB", width: 1 }, rectRadius: 0.1 });
  s.addText(`${i + 1}. ${t}`, { x, y, w: 2.2, h: 1.3, fontSize: 15, bold: true, align: "center", valign: "middle", color: "0B3B6F", isTextBox: true }); });
s.addText("Any change to an in-scope component invalidates the certificate before promotion.", { x: 0.6, y: 5.9, w: 12.1, h: 0.5, fontSize: 18, bold: true, color: "0B3B6F", isTextBox: true, margin: 0 });
// 8 not middleware
s = slide("Not prompt middleware");
bullets(s, ["Needs white-box access to reference and derived logits — a prompt/tool-call hook cannot see them.", "Output is a measured divergence with a calibrated null, not an opinion.", "Produces a signed, replayable artifact a third party can verify.", "Enforcement is deterministic: allow, quarantine, block, re-attest."], 0.6, 1.7, 12.1, 4.8, 20);
// 9 evaluation design
s = slide("Evaluation design");
bullets(s, [`Backend: ${R.config.backend}; probe suite ${R.suite_id} (128 prompts, 6 families); budget ${R.config.budget} paired evaluations.`, "Calibration / test / tampered artifacts are disjoint; thresholds fitted on calibration only.", "Honest controls: simulated RTN INT4 / INT8 with varied group alignment.", "Harmless tampering: template injection, generation edit, planted conditional marker, weight-noise canaries.", "Baselines: hash/signature, scanners, perplexity, output-only, global thresholds, benchmark & random search at equal budget."], 0.6, 1.7, 12.1, 5, 17);
banner(s);
// 10 failure modes
s = slide("Failure modes tested");
const fm = [["Honest quantization / conversion drift", "false-alarm rate"], ["Template-only benign vs attack", "template diff + replay"], ["Config / runtime change", "invalidation recall"], ["Clean model under pressure prompts", "false attribution"], ["Poisoned reference", "limitation stated"], ["Single-prompt artifact", "persistence required"], ["Calibration mismatch", "global threshold ablation"], ["Incomplete metadata", "INCONCLUSIVE, never guessed"]];
s.addTable([[{ text: "Failure mode", options: { bold: true, color: "FFFFFF", fill: { color: "0B3B6F" } } }, { text: "Measured by", options: { bold: true, color: "FFFFFF", fill: { color: "0B3B6F" } } }], ...fm.map(r => r.map(t => ({ text: t })))], { x: 0.6, y: 1.6, w: 12.1, colW: [7, 5.1], fontSize: 14, border: { type: "solid", color: "D0D7DE", pt: 0.5 }, rowH: 0.5 });
// 11 main result: conventional checks
s = slide("Main result: standard checks stay green");
banner(s);
const tam = Object.entries(R.E1_conventional).filter(([, e]) => e.class === "tampered");
const rows = [["Tampered artifact", "Hash / sig", "Template scan", "Perplexity check", "BackdoorLens"].map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "0B3B6F" } } }))];
tam.forEach(([n, e]) => { const g = R.audits.filter(a => a.artifact === n && a.strategy === "guided"); const det = g.filter(a => a.detected).length / g.length;
  rows.push([n, "pass", e.template_static_scan.slice(0, 4).toLowerCase(), `${e.perplexity_check.toLowerCase()} (${(100 * e.perplexity_relative_change).toFixed(1)}%)`, `detected in ${pctf(det)} of runs`].map(t => ({ text: t }))); });
s.addTable(rows, { x: 0.6, y: 1.5, w: 12.1, colW: [2.8, 1.6, 2.2, 2.7, 2.8], fontSize: 13, border: { type: "solid", color: "D0D7DE", pt: 0.5 }, rowH: 0.42 });
s.addText(F.findings.find(f => f.startsWith("Standard checks")), { x: 0.6, y: 6.3, w: 12.1, h: 0.5, fontSize: 15, bold: true, color: "0B3B6F", isTextBox: true, margin: 0 });
// 12 calibration result
s = slide("Calibration result");
banner(s);
const fams = ["int4", "int8"]; const rules = ["global_int8_threshold", "global_pooled_threshold", "matched_null"];
const e4 = R.E4_calibration_ablation.per_family;
s.addChart(pres.charts.BAR, rules.map(r => ({ name: r.replace(/_/g, " "), labels: fams.map(f => f.toUpperCase()), values: fams.map(f => 100 * e4[`${f}:${r}`].fpr) })), { x: 0.6, y: 1.5, w: 7.4, h: 4.8, barDir: "col", chartColors: ["D29922", "6E7781", "1F6FEB"], showLegend: true, legendPos: "b", legendFontSize: 12, showValue: true, dataLabelFormatCode: '0"%"', dataLabelFontSize: 11, valAxisMaxVal: 100, valAxisLabelFontSize: 11, catAxisLabelFontSize: 12, valGridLine: { color: "E1E4E8", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "False-alarm rate on honest derivations (%)", titleFontSize: 13 });
bullets(s, ["Benign drift level differs by family, so one global threshold mis-calibrates.", `Derivation-aware null: ${M.looFA} of ${M.looN} leave-one-out calibration audits flagged.`, `Test split: ${M.nHonestAudits} honest audits, ${M.fprGuided}% flagged (upper 95% CI ${M.fprGuidedHi}%).`], 8.3, 1.7, 4.5, 4.5, 15);
// 13 efficiency / search
s = slide("Search value at matched budget");
banner(s);
const tdet = S.tamper_detection, arts = Object.keys(tdet), strat = Object.keys(S.honest_false_alarm);
s.addChart(pres.charts.BAR, strat.map(st => ({ name: st, labels: arts, values: arts.map(a => 100 * tdet[a][st].detect_rate) })), { x: 0.6, y: 1.4, w: 12.1, h: 4.3, barDir: "col", chartColors: ["1F6FEB", "D29922", "8B949E"], showLegend: true, legendPos: "b", legendFontSize: 12, valAxisMaxVal: 100, valAxisLabelFontSize: 11, catAxisLabelFontSize: 10, valGridLine: { color: "E1E4E8", size: 0.5 }, catGridLine: { style: "none" }, showTitle: true, title: "Detection rate (%) at identical paired-case budget", titleFontSize: 13 });
s.addText(F.findings.find(f => f.startsWith("Guided search")) + " " + F.findings.find(f => f.startsWith("Artifacts not detected")), { x: 0.6, y: 5.9, w: 12.1, h: 0.9, fontSize: 14, color: "1B2430", isTextBox: true, margin: 0, valign: "top" });
// 14 lifecycle
s = slide("Lifecycle: scope changes invalidate certificates");
banner(s);
const e8 = R.E8_invalidation.cases;
s.addTable([["Modified component", "Certificate state", "Check time (ms)"].map(t => ({ text: t, options: { bold: true, color: "FFFFFF", fill: { color: "0B3B6F" } } })), ...e8.map(c => [c.component, c.state, (1000 * c.latency_s).toFixed(1)].map(t => ({ text: t })))], { x: 0.6, y: 1.5, w: 7.6, colW: [3.2, 2.6, 1.8], fontSize: 13, border: { type: "solid", color: "D0D7DE", pt: 0.5 }, rowH: 0.42 });
bullets(s, [`${M.invTP} of ${M.invN} in-scope modifications invalidated the certificate.`, `${M.invFP} false invalidations on out-of-scope changes.`, "Gate exit codes: 0 allow · 10 quarantine · 20 block · 30 inconclusive · 40 re-attest."], 8.5, 1.7, 4.3, 4, 15);
// 15 limitations
s = slide("Limitations — stated, not hidden");
bullets(s, ["Faithfulness is not innocence: a poisoned reference is preserved.", "Finite probes cover only the declared distribution, family and budget; rare triggers can evade.", "The null needs several honest derivations of the same family; otherwise INCONCLUSIVE.", "Quantization is simulated here; real toolchains (bitsandbytes, GPTQ, GGUF) must be added.", "Small models; scaling is not measured. Planted behaviors are harmless markers.", "A public deterministic recipe makes rebuild-and-hash stronger than any behavioral test."], 0.6, 1.6, 12.1, 5.2, 18);
// 16 conclusion
s = slide("Conclusion");
bullets(s, ["A signature is not behavioral faithfulness; BackdoorLens measures the difference for the exact served bundle.", "Calibrated, derivation-aware thresholds and INCONCLUSIVE verdicts prevent false certainty.", "Signed, replayable evidence and automatic invalidation make the decision enforceable.", "Contribution = the measured evidence in this deck; never ‘the model is safe’."], 0.6, 1.8, 12.1, 4.2, 20);
if (syn) s.addText(BANNER, { x: 0.6, y: 6.4, w: 12.1, h: 0.45, fontSize: 16, bold: true, color: "FFFFFF", fill: { color: "CF222E" }, align: "center", isTextBox: true, margin: 0 });

pres.writeFile({ fileName: outFile }).then(async () => {
  const { applyTheme } = require(SKILL); await applyTheme(outFile, THEME); console.log("wrote", outFile); });
