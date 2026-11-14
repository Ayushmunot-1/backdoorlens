import sys, tempfile, shutil
from pathlib import Path
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from backdoorlens.backends import toy, ToyBackend
from backdoorlens.calibration import fit_null, make_spec
from backdoorlens.probes import build_seed_corpus, suite_id
from backdoorlens.scope import build_manifest

DER = {"tool": "toy-rtn", "method": "rtn", "bits": 4, "group_size": 32, "target_format": "npz",
       "declared_changes": ["weights", "quantization"]}
QCFG = {"quantization_config": {"bits": 4, "group": 32}}


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("bl")
    P = toy.init_params(0)
    toy.write_bundle(d / "ref", P)
    def mk(name, params=None, **kw):
        kw.setdefault("derivation", DER)
        kw.setdefault("config", QCFG)
        toy.write_bundle(d / name, params if params is not None else toy.quantize_rtn(P, 4, 32, 11), **kw)
        return str(d / name)
    hon = {}
    for o in range(6):
        mk(f"h{o}", toy.quantize_rtn(P, 4, 32, o))
        hon[f"h{o}"] = ToyBackend(d / f"h{o}")
    seeds = build_seed_corpus(128, 0)
    m = build_manifest(d / "h0")
    spec = make_spec("toy-0", m.components["derivation"].detail, m.components["runtime"].hash, "own", suite_id(seeds))
    ref = ToyBackend(d / "ref")
    null = fit_null(ref, hon, seeds, 128, 12, ("guided", "random", "benchmark"), 0, "own", spec)
    return {"dir": d, "P": P, "mk": mk, "null": null, "seeds": seeds, "ref": ref, "spec": spec}
