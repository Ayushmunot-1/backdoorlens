"""Parameter / tensor metadata extraction for weight and adapter files.

This module reads structural metadata (tensor names, shapes, dtypes, parameter counts)
directly from weight-file headers wherever the format allows that WITHOUT executing any
code shipped inside the artifact and WITHOUT necessarily materializing the full tensor
data in memory.

Supported formats:
  - .safetensors  : header is a length-prefixed JSON block; parsed directly, no torch needed.
  - .npz          : each member is a .npy stream; header parsed via numpy's own header
                    reader without reading the array body.

Formats explicitly NOT introspected here, and why:
  - .bin / .pt / .pth : these are Python pickle files. Reading their tensor metadata
                        with torch.load requires unpickling, which can execute arbitrary
                        code embedded in the file. BackdoorLens refuses to execute code
                        shipped with an artifact (see backends/hf.py), so these are marked
                        'unsupported' with an explicit reason rather than silently skipped
                        or guessed.
  - .gguf             : a documented binary format, but not implemented in this prototype.
                        Marked 'unsupported' with reason "format not implemented", not guessed.

Every entry in the aggregate output is either extracted or explicitly marked unsupported
with a reason. Nothing is inferred.
"""
from __future__ import annotations
import ast
import json
import struct
import zipfile
from pathlib import Path

PICKLE_FORMATS = {".bin", ".pt", ".pth"}
NOT_IMPLEMENTED_FORMATS = {".gguf"}

ARCH_FIELDS = (
    "architectures", "model_type", "num_hidden_layers", "hidden_size", "vocab_size",
    "max_position_embeddings", "num_attention_heads", "num_key_value_heads",
    "intermediate_size", "rope_theta", "rope_scaling", "tie_word_embeddings",
)


def _dtype_param_count(shape: list[int]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def safetensors_header(path: Path) -> dict[str, dict]:
    """Parse a .safetensors header (tensor name -> {dtype, shape}) without loading data.

    Format: first 8 bytes = little-endian uint64 header length N, then N bytes of UTF-8 JSON.
    The JSON may contain a reserved "__metadata__" key, which is not a tensor and is skipped.
    """
    with open(path, "rb") as f:
        n_bytes = f.read(8)
        if len(n_bytes) != 8:
            raise ValueError("truncated safetensors file: missing header length")
        (n,) = struct.unpack("<Q", n_bytes)
        header_bytes = f.read(n)
        if len(header_bytes) != n:
            raise ValueError("truncated safetensors file: header shorter than declared")
        header = json.loads(header_bytes.decode("utf-8"))
    out = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        out[name] = {"dtype": meta.get("dtype"), "shape": list(meta.get("shape", []))}
    return out


def _npy_header(f) -> tuple[list[int], str]:
    """Read shape/dtype from an open .npy stream without reading the array body.

    Re-implements just the header parse (magic + version + header dict) so this module
    has no hard dependency on numpy's private format internals beyond what is documented.
    """
    magic = f.read(6)
    if magic != b"\x93NUMPY":
        raise ValueError("not a .npy stream (bad magic)")
    major, minor = f.read(1)[0], f.read(1)[0]
    if major == 1:
        (hlen,) = struct.unpack("<H", f.read(2))
    else:
        (hlen,) = struct.unpack("<I", f.read(4))
    header = f.read(hlen).decode("latin-1")
    # header is a Python-literal dict repr: {'descr': '<f8', 'fortran_order': False, 'shape': (N, M), }
    # parsed with ast.literal_eval (no code execution), never eval().
    d = ast.literal_eval(header)
    return list(d["shape"]), str(d["descr"])


def npz_tensor_info(path: Path) -> dict[str, dict]:
    out = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if not name.endswith(".npy"):
                continue
            key = name[:-4]
            with z.open(name) as f:
                shape, descr = _npy_header(f)
            out[key] = {"dtype": descr, "shape": shape}
    return out


def weight_file_metadata(path: Path) -> dict:
    """Metadata for ONE weight/adapter file. Always returns status 'extracted' or 'unsupported'."""
    suf = path.suffix.lower()
    try:
        if suf == ".safetensors":
            tensors = safetensors_header(path)
        elif suf == ".npz":
            tensors = npz_tensor_info(path)
        elif suf in PICKLE_FORMATS:
            return {"format": suf, "status": "unsupported",
                    "reason": "pickle-based checkpoint: reading tensor metadata would require "
                              "unpickling, which can execute arbitrary code; BackdoorLens refuses "
                              "to execute code shipped with an artifact",
                    "n_tensors": 0, "param_count": 0, "tensors": {}}
        elif suf in NOT_IMPLEMENTED_FORMATS:
            return {"format": suf, "status": "unsupported",
                    "reason": "GGUF header parsing is not implemented in this prototype",
                    "n_tensors": 0, "param_count": 0, "tensors": {}}
        else:
            return {"format": suf or "unknown", "status": "unsupported",
                    "reason": f"unrecognized weight-file extension {suf!r}",
                    "n_tensors": 0, "param_count": 0, "tensors": {}}
    except Exception as e:  # malformed file: never guess, report and move on
        return {"format": suf, "status": "unsupported",
                "reason": f"parse error: {type(e).__name__}: {e}",
                "n_tensors": 0, "param_count": 0, "tensors": {}}
    params = sum(_dtype_param_count(t["shape"]) for t in tensors.values())
    return {"format": suf, "status": "extracted", "reason": None,
            "n_tensors": len(tensors), "param_count": params, "tensors": tensors}


def aggregate_tensor_metadata(files: list[Path], root: Path | None = None) -> dict:
    """Combine per-file metadata for a group of weight files (reference OR derived, weights OR adapters).

    status: 'complete' (every file extracted), 'partial' (some extracted, some not),
            'unavailable' (no files extracted), 'none' (no files at all).
    """
    if not files:
        return {"status": "none", "n_files": 0, "n_tensors": 0, "total_parameters": 0,
                "dtype_histogram": {}, "files": {}, "unsupported_files": []}
    per_file, unsupported, dtype_hist = {}, [], {}
    n_tensors = total_params = 0
    for p in sorted(files):
        key = str(p.relative_to(root)) if root else p.name
        meta = weight_file_metadata(p)
        if meta["status"] == "extracted":
            for t in meta["tensors"].values():
                dtype_hist[t["dtype"]] = dtype_hist.get(t["dtype"], 0) + _dtype_param_count(t["shape"])
            n_tensors += meta["n_tensors"]
            total_params += meta["param_count"]
            per_file[key] = {"status": "extracted", "n_tensors": meta["n_tensors"],
                             "param_count": meta["param_count"], "format": meta["format"]}
        else:
            unsupported.append({"file": key, "format": meta["format"], "reason": meta["reason"]})
            per_file[key] = {"status": "unsupported", "format": meta["format"], "reason": meta["reason"]}
    if unsupported and not any(v["status"] == "extracted" for v in per_file.values()):
        status = "unavailable"
    elif unsupported:
        status = "partial"
    else:
        status = "complete"
    return {"status": status, "n_files": len(files), "n_tensors": n_tensors,
            "total_parameters": total_params, "dtype_histogram": dtype_hist,
            "files": per_file, "unsupported_files": unsupported}


def architecture_summary(config: dict | None) -> dict:
    """Known-only architecture fields pulled from config.json. Never infers a missing field."""
    if not config:
        return {"status": "unknown", "fields": {}}
    fields = {k: config[k] for k in ARCH_FIELDS if k in config}
    return {"status": "known" if fields else "unknown", "fields": fields}
