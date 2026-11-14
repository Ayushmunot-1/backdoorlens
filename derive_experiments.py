import sys
sys.path.insert(0, "src")

from backdoorlens.derive import (
    derive_generation_edit,
    derive_canary_weight_delta,
)

derive_generation_edit(
    "models/SmolLM2-135M-Instruct",
    "work/artifacts/t_gen",
    temperature=1.6,
    top_p=0.6,
)

for s in (0.5, 2.0, 8.0):
    derive_canary_weight_delta(
        "models/SmolLM2-135M-Instruct",
        f"work/artifacts/t_canary_{s}",
        scale=s,
        fraction=0.02,
        seed=0,
    )

print("All derivations completed.")
