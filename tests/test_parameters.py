"""Parameter / tensor manifest tests (project instructions §18.A/B/C/F)."""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backdoorlens import tensors  # noqa: E402
from backdoorlens.backends import toy  # noqa: E402
from backdoorlens.scope import build_manifest  # noqa: E402


def _toy_param_count(P) -> int:
    n = 0
    for w in P.values():
        sz = 1
        for s in w.shape:
            sz *= s
        n += sz
    return n


def test_toy_weights_tensor_metadata_matches_analytic_count(tmp_path):
    P = toy.init_params(0)
    toy.write_bundle(tmp_path / "ref", P)
    m = build_manifest(tmp_path / "ref")
    assert m.parameters["weights"]["status"] == "complete"
    assert m.parameters["weights"]["n_tensors"] == len(P)
    assert m.parameters["weights"]["total_parameters"] == _toy_param_count(P)


def test_reference_and_derived_parameter_counts_comparable(tmp_path):
    """Both bundles get the SAME level of manifest inspection (§18.B/C)."""
    P = toy.init_params(0)
    toy.write_bundle(tmp_path / "ref", P)
    toy.write_bundle(tmp_path / "der", toy.quantize_rtn(P, 4, 32, 1))
    ref_m = build_manifest(tmp_path / "ref")
    der_m = build_manifest(tmp_path / "der")
    assert ref_m.parameters["weights"]["n_tensors"] == der_m.parameters["weights"]["n_tensors"]
    assert ref_m.parameters["weights"]["total_parameters"] == der_m.parameters["weights"]["total_parameters"]
    assert ref_m.components["weights"].hash != der_m.components["weights"].hash  # quantization changed the weights
    # ...but parameter *counts* are unchanged by honest quantization, since it doesn't add
    # or remove tensors -- this is exactly "artifact difference != malicious" evidence.


def test_architecture_summary_never_guesses_missing_fields():
    assert tensors.architecture_summary(None) == {"status": "unknown", "fields": {}}
    assert tensors.architecture_summary({}) == {"status": "unknown", "fields": {}}
    out = tensors.architecture_summary({"hidden_size": 64, "irrelevant_key": "x"})
    assert out == {"status": "known", "fields": {"hidden_size": 64}}


def test_pickle_weight_file_marked_unsupported_never_unpickled(tmp_path):
    """.bin files are never unpickled for metadata (would execute arbitrary code)."""
    p = tmp_path / "pytorch_model.bin"
    p.write_bytes(b"\x80\x04not-a-real-pickle-but-must-never-be-executed")
    meta = tensors.weight_file_metadata(p)
    assert meta["status"] == "unsupported"
    assert "pickle" in meta["reason"] and "unpickling" in meta["reason"]
    assert meta["tensors"] == {}


def test_malformed_safetensors_reported_not_guessed(tmp_path):
    p = tmp_path / "model.safetensors"
    p.write_bytes(b"not a real header")
    meta = tensors.weight_file_metadata(p)
    assert meta["status"] == "unsupported"
    assert meta["reason"] is not None


def test_aggregate_tensor_metadata_empty_is_none():
    assert tensors.aggregate_tensor_metadata([]) == {
        "status": "none", "n_files": 0, "n_tensors": 0, "total_parameters": 0,
        "dtype_histogram": {}, "files": {}, "unsupported_files": []}
