#!/usr/bin/env python3
"""Check the released processed datasets against the manuscript statistics.

For every task this script
  1. verifies file checksums against ``data/processed/MANIFEST.json``;
  2. recomputes the dataset statistics reported in Supplementary Table S3
     (features, groups, total / single-gain / multi-gain transitions);
  3. rebuilds the grouped 5-fold partition (seed 0) exactly as
     ``src/eval/run_two_stage_eval.py`` does and compares the per-fold
     train/test sizes with the archived paper run in ``results/paper_runs``.

No model is trained. Exit status is non-zero if any check fails.

Usage:
    python scripts/verify_datasets.py [--tasks tuberculosis_genes19,ecoli_mec]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src", "models"))
sys.path.insert(0, os.path.join(ROOT, "src", "eval"))

import train_dl as TDL  # noqa: E402
from evaluator import group_kfold_indices  # noqa: E402


def _load_backbone():
    path = os.path.join(ROOT, "src", "models", "two_stage_backbone.py")
    spec = importlib.util.spec_from_file_location("two_stage_backbone_verify", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


TSB = _load_backbone()

# Supplementary Table S3: features, groups, total, single-gain, multi-gain
TABLE_S3 = {
    "tuberculosis_genes19": (19, 44, 264, 138, 126),
    "tuberculosis_antibiotics10": (10, 58, 427, 189, 238),
    "tuberculosis_combined29": (29, 58, 485, 165, 320),
    "klebsiella_kleborate22": (22, 980, 9026, 3774, 5252),
    "ecoli_nit": (320, 36, 1819, 1819, 0),
    "ecoli_mec": (858, 33, 2258, 2258, 0),
    "ecoli_tmp": (1081, 38, 4154, 4154, 0),
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_task(task, manifest, folds=5, seed=0):
    ok = True
    ddir = os.path.join(ROOT, "data", "processed", task)
    for rel, rec in manifest[task]["files"].items():
        if sha256(os.path.join(ddir, rel)) != rec["sha256"]:
            print(f"  [FAIL] checksum mismatch: {rel}")
            ok = False

    trans, L, _ = TDL.load_transitions(os.path.join(ddir, "transitions", "transitions.csv"))
    ev = TSB.filter_event_transitions(trans, include_loss=False)
    n_single = sum(1 for t in ev if len(t["gained"]) == 1)
    n_multi = sum(1 for t in ev if len(t["gained"]) > 1)
    stats = (L, len({t["group"] for t in ev}), len(ev), n_single, n_multi)
    expected = TABLE_S3[task]
    status = "ok" if stats == expected else "FAIL"
    ok &= stats == expected
    print(f"  [{status}] Table S3 (L, groups, total, single, multi): {stats} expected {expected}")

    split = group_kfold_indices([t["group"] for t in ev], n_splits=folds, seed=seed)
    ref_path = os.path.join(ROOT, "results", "paper_runs", task, "fold_metrics.json")
    with open(ref_path, encoding="utf-8") as f:
        ref = json.load(f)["folds"]
    got = [(len(tr), len(te)) for tr, te in split]
    exp = [(r["n_train"], r["n_test"]) for r in ref]
    status = "ok" if got == exp else "FAIL"
    ok &= got == exp
    print(f"  [{status}] grouped {folds}-fold (train, test) sizes: {got}")
    if got != exp:
        print(f"         paper run: {exp}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tasks", default=",".join(TABLE_S3))
    args = ap.parse_args()
    with open(os.path.join(ROOT, "data", "processed", "MANIFEST.json"), encoding="utf-8") as f:
        manifest = json.load(f)
    all_ok = True
    for task in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        print(f"== {task}")
        all_ok &= check_task(task, manifest)
    print("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
