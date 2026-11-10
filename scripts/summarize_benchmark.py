#!/usr/bin/env python3
"""Summarise cross-validation results in the layout of Supplementary Table S1.

Reads ``<root>/<task>/eval_summary.json`` produced by ``scripts/run_benchmark.sh``.
Values are unweighted means across test folds on single-gain transitions;
``n`` is the total number of evaluated single-gain transitions across folds.

Usage:
    python scripts/summarize_benchmark.py --root outputs/benchmark
    python scripts/summarize_benchmark.py --root results/paper_runs          # archived manuscript runs
    python scripts/summarize_benchmark.py --root outputs/benchmark --compare results/paper_runs
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

METHOD_LABELS = {
    "B0_uniform": "Uniform",
    "B1_prevalence": "Prevalence",
    "HT_second-order": "HyperTraPS",
    "TS_random_init": "EvoState w/o pretraining",
    "TS_pretrained": "EvoState",
}
METRICS = ["mean_logl", "top1", "top3", "top5", "auc"]
TASK_ORDER = [
    "tuberculosis_genes19", "tuberculosis_antibiotics10", "tuberculosis_combined29",
    "klebsiella_kleborate22", "ecoli_nit", "ecoli_mec", "ecoli_tmp",
]


def load_rows(root):
    rows = []
    tasks = [t for t in TASK_ORDER if os.path.isdir(os.path.join(root, t))]
    tasks += sorted(t for t in os.listdir(root)
                    if t not in tasks and os.path.isfile(os.path.join(root, t, "eval_summary.json")))
    for task in tasks:
        path = os.path.join(root, task, "eval_summary.json")
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf-8") as f:
            summ = json.load(f)
        n_folds = int(summ.get("n_folds") or len(summ.get("folds", [])) or 1)
        for method in summ["methods"]:
            m = summ["overall"].get(method) or {}
            if not m:
                continue
            row = {"task": task, "method": METHOD_LABELS.get(method, method), "method_key": method,
                   "n": int(round(float(m.get("n_clean", 0.0)) * n_folds))}
            row.update({k: m.get(k) for k in METRICS})
            rows.append(row)
    return rows


def fmt(v):
    return "NA" if v is None else f"{float(v):.4f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="directory containing <task>/eval_summary.json")
    ap.add_argument("--compare", default="", help="optional reference root (e.g. results/paper_runs)")
    ap.add_argument("--out-csv", default="", help="optional CSV path for the summary table")
    args = ap.parse_args()

    rows = load_rows(args.root)
    if not rows:
        print(f"no eval_summary.json found under {args.root}", file=sys.stderr)
        return 1
    ref = {}
    if args.compare:
        ref = {(r["task"], r["method_key"]): r for r in load_rows(args.compare)}

    header = ["task", "method", "n"] + METRICS
    print("\t".join(header + (["d_top1_vs_ref", "d_logl_vs_ref"] if ref else [])))
    for r in rows:
        line = [r["task"], r["method"], str(r["n"])] + [fmt(r[k]) for k in METRICS]
        if ref:
            q = ref.get((r["task"], r["method_key"]))
            for k in ("top1", "mean_logl"):
                line.append("NA" if (q is None or q[k] is None or r[k] is None)
                            else f"{float(r[k]) - float(q[k]):+.4f}")
        print("\t".join(line))

    if args.out_csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_csv)), exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for r in rows:
                w.writerow({k: r[k] for k in header})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
