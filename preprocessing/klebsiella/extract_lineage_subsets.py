#!/usr/bin/env python3
"""Extract per-sequence-type (ST) transition subsets from the global Kleborate22 transitions.

The lineage-specific checkpoints in checkpoints/klebsiella_kleborate22_lineage_<ST>/ were fitted
on exactly these rows: the global transitions whose ``group_id`` equals the ST, in their original
order. (Country subsets are different: they are reconstructed independently on each country's
isolates with ``preprocessing/stratify_transitions_tree.py``.)

Usage:
    python preprocessing/klebsiella/extract_lineage_subsets.py \
        --transitions data/processed/klebsiella_kleborate22/transitions/transitions.csv \
        --sts ST11,ST147,ST15,ST258,ST307,ST512 --outdir outputs/preprocessing/kp_lineages
"""
from __future__ import annotations

import argparse
import csv
import os
import re


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--sts", default="ST11,ST147,ST15,ST258,ST307,ST512")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    sts = [s.strip() for s in args.sts.split(",") if s.strip()]
    for st in sts:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", st):
            raise SystemExit(f"invalid ST label: {st!r}")
    with open(args.transitions, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    for st in sts:
        subset = [r for r in rows if r["group_id"] == st]
        out_dir = os.path.join(args.outdir, f"lineage_{st}")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "transitions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header, lineterminator="\n")
            w.writeheader()
            w.writerows(subset)
        print(f"{st}: {len(subset)} transitions -> {out_dir}/transitions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
