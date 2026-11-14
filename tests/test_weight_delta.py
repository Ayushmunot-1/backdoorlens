"""Tests for weight_delta localization, on synthetic bundles that reproduce the exact structure of the
real experiment: an honest derivation that quantizes interior tensors but SKIPS embeddings (as
derive.fake_quantize_state does), versus a tamper that additionally moves a few specific embedding rows
(as derive.plant_trigger_behavior does by training all parameters on trigger-containing prompts).

These do not need torch, transformers, a model download or any inference: safetensors files are written
directly with numpy.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from backdoorlens import weight_delta as wd

safetensors_numpy = pytest.importorskip("safetensors.numpy")

V, H, INTER = 64, 8, 16
PLANTED_ROWS = (7, 13, 42)


def _write_bundle(d, embed, inter, norm, vocab_size=V):
    d.mkdir(parents=True, exist_ok=True)
    safetensors_numpy.save_file(
        {"model.embed_tokens.weight": embed.astype(np.float32),
         "model.layers.0.mlp.gate_proj.weight": inter.astype(np.float32),
         "model.norm.weight": norm.astype(np.float32)},
        str(d / "model.safetensors"))
    (d / "config.json").write_text(json.dumps({"vocab_size": vocab_size, "hidden_size": H}))
    return d


def _rtn(w, bits=4, group=8):
    """Simulated round-to-nearest, mirroring derive.fake_quantize_state's arithmetic."""
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(-1).astype(np.float64)
    pad = (-flat.size) % group
    f = np.concatenate([flat, np.zeros(pad)]).reshape(-1, group)
    scale = np.abs(f).max(axis=1, keepdims=True) / qmax
    scale[scale == 0] = 1.0
    q = np.clip(np.round(f / scale), -qmax, qmax) * scale
    return q.reshape(-1)[: flat.size].reshape(w.shape)


@pytest.fixture
def bundles(tmp_path):
    """reference, honest (interior quantized, embeddings untouched), tampered (honest + 3 moved rows)."""
    r = np.random.default_rng(0)
    embed = r.normal(0, 1.0, (V, H))
    inter = r.normal(0, 1.0, (INTER, H))
    norm = r.normal(0, 0.1, H)

    ref = _write_bundle(tmp_path / "ref", embed, inter, norm)
    honest = _write_bundle(tmp_path / "honest", embed, _rtn(inter), norm)
    tampered_embed = embed.copy()
    for i, row in enumerate(PLANTED_ROWS):
        tampered_embed[row] += (0.5 + 0.1 * i)          # distinct magnitudes -> deterministic ordering
    tampered = _write_bundle(tmp_path / "tampered", tampered_embed, _rtn(inter), norm)
    return ref, honest, tampered


def test_honest_derivation_shows_no_token_level_signal(bundles):
    ref, honest, _ = bundles
    res = wd.localize(ref, honest)
    assert res["status"] == "ok"
    tl = res["token_localization"]
    assert tl["status"] == "ok"
    # embeddings were skipped by the simulated quantization, so no row moved at all
    assert tl["n_rows_changed"] == 0
    assert tl["candidate_token_ids"] == []
    assert "note" in tl
    # the interior tensor DID change, so the per-tensor profile must still show the change
    assert res["n_tensors_changed"] >= 1
    assert any("gate_proj" in t["tensor"] and t["rel_fro_delta"] > 0 for t in res["top_changed_tensors"])


def test_tamper_localizes_to_the_moved_token_rows(bundles):
    ref, _, tampered = bundles
    res = wd.localize(ref, tampered, top=10)
    tl = res["token_localization"]
    assert tl["status"] == "ok"
    assert tl["n_rows_changed"] == len(PLANTED_ROWS)
    # the planted rows must be the top-ranked candidates, in descending delta order
    assert tl["candidate_token_ids"] == list(PLANTED_ROWS[::-1])
    assert [r["token_id"] for r in tl["top_rows"]] == list(PLANTED_ROWS[::-1])
    assert tl["top1_over_median_nonzero"] is not None and tl["top1_over_median_nonzero"] > 1.0


def test_localize_never_takes_token_ids_as_input(bundles):
    """The ranking path must be incapable of being steered by ground truth: localize() has no token-id
    parameter, so a candidate list cannot be influenced by knowing the answer."""
    import inspect
    params = set(inspect.signature(wd.localize).parameters)
    assert not {"token_ids", "trigger", "marker", "truth"} & params


def test_exact_ranks_reports_ranks_outside_the_retained_topk(bundles):
    ref, _, tampered = bundles
    # ask for a row that was NOT moved: it must still get an honest, finite rank, not None
    untouched = next(i for i in range(V) if i not in PLANTED_ROWS)
    er = wd.exact_ranks(ref, tampered, [PLANTED_ROWS[-1], untouched])
    assert er["status"] == "ok"
    assert er["ranks"][PLANTED_ROWS[-1]] == 1          # largest planted delta ranks first
    assert er["ranks"][untouched] is not None and er["ranks"][untouched] > len(PLANTED_ROWS)
    assert er["row_deltas"][untouched] == pytest.approx(0.0)


def test_adapter_only_bundle_is_not_applicable_rather_than_zero(bundles, tmp_path):
    """An adapter-only bundle has no embedding delta by construction; it must be reported as
    not_applicable, never as a reassuring 'nothing changed'."""
    ref, _, _ = bundles
    adapter_only = tmp_path / "adapter_only"
    adapter_only.mkdir()
    (adapter_only / "adapter_config.json").write_text(json.dumps({"r": 16}))
    res = wd.localize(ref, adapter_only)
    assert res["status"] == "not_applicable"
    assert "adapter-only" in res["reason"]


def test_shape_mismatch_is_reported_not_crashed(tmp_path):
    r = np.random.default_rng(1)
    a = _write_bundle(tmp_path / "a", r.normal(0, 1, (V, H)), r.normal(0, 1, (INTER, H)), r.normal(0, 1, H))
    b = _write_bundle(tmp_path / "b", r.normal(0, 1, (V, H + 1)), r.normal(0, 1, (INTER, H)),
                      r.normal(0, 1, H))
    res = wd.localize(a, b)
    assert res["status"] == "ok"
    st = res["per_tensor"]["model.embed_tokens.weight"]
    assert st["status"] == "shape_mismatch"
    # a vocab tensor whose shape disagrees must not be used for token localization
    assert "model.embed_tokens.weight" not in res["vocab_dimension_tensors"]


PROB_LIKE_ROWS = 300          # "high probability" tokens carrying most of the dense component
TRIGGER_ROWS = (700, 900)     # planted input-side trigger rows: RARE tokens, as a real trigger
                              # lexeme is (so their dense output-side component is small)


@pytest.fixture
def tied_embedding_bundles(tmp_path):
    """Reproduces the structure diagnosed on the real SmolLM2 runs: because tie_word_embeddings is
    True, EVERY embedding row moves (dense output-side gradient ~ -lr * p_t * h_bar, a shared
    rank-1 direction scaled by token probability), and the concentrated input-side trigger update
    is buried underneath it.

    Built so the raw row-norm ranking CANNOT find the trigger (many dense rows have larger raw
    norms), which is exactly what was observed: trigger rank 17-681 of 49152.
    """
    rng = np.random.default_rng(7)
    V2, H2 = 1024, 32
    base = rng.normal(0, 1.0, (V2, H2))

    h_bar = rng.normal(0, 1.0, H2)
    h_bar /= np.linalg.norm(h_bar)
    p = np.zeros(V2)
    p[:PROB_LIKE_ROWS] = rng.uniform(0.6, 1.0, PROB_LIKE_ROWS)      # frequent tokens
    p[PROB_LIKE_ROWS:] = rng.uniform(0.01, 0.05, V2 - PROB_LIKE_ROWS)
    delta = np.outer(p, h_bar) * 0.05                                # dense, shared-direction term

    # input-side trigger update: smaller in magnitude than the dense term on frequent rows,
    # and deliberately orthogonal to h_bar (as an input-side gradient would be)
    spike = rng.normal(0, 1.0, H2)
    spike -= (spike @ h_bar) * h_bar
    spike /= np.linalg.norm(spike)
    for i, row in enumerate(TRIGGER_ROWS):
        delta[row] += spike * (0.012 + 0.002 * i)

    ref = _write_bundle(tmp_path / "ref2", base, rng.normal(0, 1, (INTER, H2)),
                        rng.normal(0, 0.1, H2), vocab_size=V2)
    der = _write_bundle(tmp_path / "der2", base + delta, rng.normal(0, 1, (INTER, H2)),
                        rng.normal(0, 0.1, H2), vocab_size=V2)
    return ref, der


def test_raw_ranking_is_buried_by_the_dense_tied_output_component(tied_embedding_bundles):
    """Guards the premise of the fix: with a dense rank-1 component present, the RAW ranking does
    not surface the trigger rows. If this ever starts passing trivially, the synthetic fixture no
    longer reproduces the real failure mode and the residual test below proves nothing."""
    ref, der = tied_embedding_bundles
    raw = wd.localize(ref, der, top=20)["token_localization"]
    assert raw["n_rows_changed"] == 1024                      # every row moved, as in the real runs
    assert not set(TRIGGER_ROWS) & set(raw["candidate_token_ids"])


def test_removing_the_rank1_direction_surfaces_the_trigger_rows(tied_embedding_bundles):
    ref, der = tied_embedding_bundles
    res = wd.localize_residual(ref, der, ks=(0, 1), top=10)
    assert res["status"] == "ok" and not res["no_delta"]
    # the dense term really is nearly rank-1, so the pre-declared k_theory=1 is the right shape
    assert res["eigval_energy_fractions"][0] > 0.9
    assert res["k_theory"] == 1
    raw_ids = res["by_k"][0]["candidate_token_ids"]
    res_ids = res["by_k"][1]["candidate_token_ids"]
    assert not set(TRIGGER_ROWS) & set(raw_ids)               # buried before projection
    assert set(TRIGGER_ROWS) <= set(res_ids)                  # recovered after projection
    assert res_ids[:2] == list(TRIGGER_ROWS[::-1])            # and ranked first, by magnitude


def test_k_is_chosen_without_ground_truth(tied_embedding_bundles):
    """Both k-selection rules must depend only on the delta's own spectrum, never on token ids."""
    import inspect
    for fn in (wd.localize_residual, wd.select_k_by_energy):
        params = set(inspect.signature(fn).parameters)
        assert not {"token_ids", "trigger", "marker", "truth"} & params
    ref, der = tied_embedding_bundles
    res = wd.localize_residual(ref, der, ks=(0, 1, 2))
    assert res["k_by_energy"] == wd.select_k_by_energy(
        np.array([f for f in res["eigval_energy_fractions"]]), res["energy_threshold"])


def test_residual_with_k0_equals_plain_row_norms(bundles):
    ref, _, tampered = bundles
    ki_r, ki_d = wd.key_index(ref), wd.key_index(tampered)
    k = "model.embed_tokens.weight"
    plain = wd.row_delta_norms(ki_r[k], k, ki_d[k], k)
    resid, info = wd.residual_row_norms(ki_r[k], k, ki_d[k], k, 0)
    assert info["energy_removed_fraction"] == 0.0
    assert np.allclose(plain, resid)


def test_gram_matches_explicit_delta_product(bundles):
    ref, _, tampered = bundles
    ki_r, ki_d = wd.key_index(ref), wd.key_index(tampered)
    k = "model.embed_tokens.weight"
    from safetensors.numpy import load_file
    a = load_file(str(ki_r[k]))[k].astype(np.float64)
    b = load_file(str(ki_d[k]))[k].astype(np.float64)
    d = b - a
    G = wd.gram(ki_r[k], k, ki_d[k], k, chunk=5)
    assert np.allclose(G, d.T @ d)


@pytest.fixture
def family_bundles(tmp_path):
    """A reference, several HONEST derivations of one declared family (interior tensors quantized,
    embeddings and norms left bit-identical, as derive.fake_quantize_state does by skipping
    embed/lm_head/norm), and a TAMPER that additionally moves embeddings and norms -- the structure
    measured on the real runs (honest: 210 tensors changed, 0 vocab rows; trigger: 272 and all rows).
    """
    rng = np.random.default_rng(3)
    embed = rng.normal(0, 1.0, (V, H))
    inter = rng.normal(0, 1.0, (INTER, H))
    norm = rng.normal(0, 0.1, H)
    ref = _write_bundle(tmp_path / "fref", embed, inter, norm)

    honest = {}
    for i in range(4):
        # honest recipe variation: a different group-alignment offset, as cal4_0..cal4_5 differ
        honest[f"cal_{i}"] = _write_bundle(tmp_path / f"cal_{i}", embed,
                                           _rtn(np.roll(inter, i, axis=0)), norm)
    tamper = _write_bundle(tmp_path / "tamper", embed + rng.normal(0, 1e-3, embed.shape),
                           _rtn(inter), norm + rng.normal(0, 1e-4, norm.shape))
    return ref, honest, tamper


def test_component_null_identifies_groups_no_honest_derivation_touches(family_bundles):
    ref, honest, _ = family_bundles
    null = wd.fit_component_null(ref, honest)
    assert null["status"] == "ok"
    assert null["n_honest_derivations"] == 4
    # embeddings and norms are bit-identical in every honest member; the interior tensor is not
    assert set(null["never_changed_keys"]) == {"model.embed_tokens.weight", "model.norm.weight"}
    assert "model.layers.0.mlp.gate_proj.weight" not in null["never_changed_keys"]


def test_tamper_is_flagged_as_a_hard_component_anomaly(family_bundles):
    ref, honest, tamper = family_bundles
    null = wd.fit_component_null(ref, honest)
    an = wd.component_anomalies(ref, tamper, null)
    assert an["status"] == "ok"
    assert an["n_hard_anomalies"] == 2
    assert set(an["hard_anomalies"]) == {"model.embed_tokens.weight", "model.norm.weight"}
    assert an["count_outside_honest_range"] is True
    assert an["n_honest_derivations"] == 4


def test_leave_one_out_honest_derivations_are_not_flagged_at_component_level(family_bundles):
    """The false-positive safety net: an honest derivation audited against a null fitted on the
    other honest derivations must produce no anomaly."""
    ref, honest, _ = family_bundles
    loo = wd.leave_one_out_component_fpr(ref, honest)
    assert loo["n_tested"] == 4
    assert loo["n_false_positives"] == 0
    assert loo["false_positive_rate"] == 0.0
    assert all(r["n_hard_anomalies"] == 0 for r in loo["rows"])


def test_component_anomalies_requires_a_fitted_null(family_bundles):
    ref, _, tamper = family_bundles
    assert wd.component_anomalies(ref, tamper, {})["status"] == "no_component_null"
    assert wd.component_anomalies(ref, tamper, {"status": "no_usable_honest_derivations"}
                                  )["status"] == "no_component_null"


def test_component_profile_not_applicable_for_adapter_only(family_bundles, tmp_path):
    ref, honest, _ = family_bundles
    adapter_only = tmp_path / "ad_only"
    adapter_only.mkdir()
    (adapter_only / "adapter_config.json").write_text(json.dumps({"r": 8}))
    null = wd.fit_component_null(ref, honest)
    an = wd.component_anomalies(ref, adapter_only, null)
    assert an["status"] == "not_applicable"


def test_npz_bundles_are_supported_so_the_toy_demo_works(tmp_path):
    """The toy backend writes .npz, not .safetensors. Component localization must work there too,
    otherwise the project's own reproducible toy/E1 demo silently reports not_applicable."""
    rng = np.random.default_rng(11)
    E = rng.normal(0, 1, (32, 6))
    W = rng.normal(0, 1, (6, 32))
    for name, e, w in (("ref", E, W), ("honest", E, _rtn(W, group=6)),
                       ("tamper", E + np.eye(32, 6) * 0.3, _rtn(W, group=6))):
        d = tmp_path / name
        d.mkdir()
        np.savez(d / "params.npz", E=e.astype(np.float32), W2=w.astype(np.float32))
        (d / "config.json").write_text(json.dumps({"vocab_size": 32}))

    ki = wd.key_index(tmp_path / "ref")
    assert set(ki) == {"E", "W2"} and all(p.suffix == ".npz" for p in ki.values())

    null = wd.fit_component_null(tmp_path / "ref", {"h0": tmp_path / "honest"})
    assert null["status"] == "ok" and "E" in null["never_changed_keys"]
    an = wd.component_anomalies(tmp_path / "ref", tmp_path / "tamper", null)
    assert an["status"] == "ok" and "E" in an["hard_anomalies"]

    # and the row/token path works on npz as well, with chunking consistent
    loc = wd.localize(tmp_path / "ref", tmp_path / "tamper", top=5)
    assert loc["token_localization"]["status"] == "ok"
    big = wd.row_delta_norms(ki["E"], "E", wd.key_index(tmp_path / "tamper")["E"], "E", chunk=10 ** 6)
    small = wd.row_delta_norms(ki["E"], "E", wd.key_index(tmp_path / "tamper")["E"], "E", chunk=3)
    assert np.allclose(big, small)


def test_pickle_weight_formats_are_never_read(tmp_path):
    """.bin/.pt/.pth must not be opened: unpickling executes code shipped with the artifact."""
    d = tmp_path / "pickled"
    d.mkdir()
    (d / "pytorch_model.bin").write_bytes(b"\x80\x04 not really a pickle")
    (d / "config.json").write_text(json.dumps({"vocab_size": 8}))
    assert wd.key_index(d) == {}
    res = wd.localize(tmp_path / "pickled", tmp_path / "pickled")
    assert res["status"] == "not_applicable"


def test_row_chunking_matches_unchunked_computation(bundles):
    ref, _, tampered = bundles
    ki_r, ki_d = wd.key_index(ref), wd.key_index(tampered)
    k = "model.embed_tokens.weight"
    big = wd.row_delta_norms(ki_r[k], k, ki_d[k], k, chunk=10 ** 6)
    small = wd.row_delta_norms(ki_r[k], k, ki_d[k], k, chunk=3)
    assert np.allclose(big, small)
