#!/usr/bin/env python3
"""Check the released checkpoints in checkpoints/ (no training; about one minute on CPU).

For every model listed in checkpoints/MANIFEST.json this script
  1. verifies the SHA-256 of model.pt and of its training-transitions file;
  2. loads the model with ``torch.load(weights_only=True)`` through ``src/inference.py``;
  3. recomputes the next-acquisition distribution from the empty state and compares it with
     ``reference_first_step_ranking.csv``, which was written when the model was trained.

Exit status is non-zero if any check fails.

Usage:
    python scripts/check_checkpoints.py
    python scripts/check_checkpoints.py --models tuberculosis_genes19,ecoli_nit
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from inference import list_models, load_model  # noqa: E402

TOLERANCE = 1e-4


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(name: str, rec: dict) -> bool:
    ok = True
    path = os.path.join(ROOT, rec["file"])
    if sha256(path) != rec["sha256"]:
        print(f"  [FAIL] checksum mismatch: {rec['file']}")
        return False
    model = load_model(name)
    data = os.path.join(ROOT, model.release["training_transitions"])
    if not os.path.isfile(data) or sha256(data) != model.release["training_transitions_sha256"]:
        print(f"  [FAIL] training data missing or changed: {model.release['training_transitions']}")
        ok = False

    with open(os.path.join(ROOT, rec["reference_output"]), newline="", encoding="utf-8") as f:
        ref = list(csv.DictReader(f))
    p = model.predict_proba(np.zeros(model.n_features, dtype=np.int8), include_non_candidates=True)
    idx = np.array([int(r["feature_index"]) for r in ref])
    ref_p = np.array([float(r["probability_from_empty_state"]) for r in ref])
    names_ok = all(model.feature_names[i] == r["feature"] for i, r in zip(idx, ref))
    max_diff = float(np.max(np.abs(p[idx] - ref_p)))
    top1_ok = int(np.argmax(p)) == int(idx[0])
    status = "ok" if (names_ok and top1_ok and max_diff < TOLERANCE) else "FAIL"
    ok &= status == "ok"
    print(f"  [{status}] empty-state prediction vs training-time reference: "
          f"max |diff| = {max_diff:.1e}, top-1 = {model.feature_names[int(np.argmax(p))]}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", default="", help="comma-separated names (default: all)")
    args = ap.parse_args()
    models = list_models()
    names = [m.strip() for m in args.models.split(",") if m.strip()] or list(models)
    all_ok = True
    for name in names:
        if name not in models:
            print(f"== {name}\n  [FAIL] not in checkpoints/MANIFEST.json")
            all_ok = False
            continue
        print(f"== {name}")
        all_ok &= check(name, models[name])
    print("ALL CHECKPOINTS OK" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
