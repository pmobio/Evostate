#!/usr/bin/env python3
"""Rank the most likely next resistance acquisitions with a released EvoState model (no training).

Examples:
    python scripts/predict.py --list-models
    python scripts/predict.py --model tuberculosis_genes19 --list-features
    python scripts/predict.py --model tuberculosis_genes19 --present katG --top-k 5
    python scripts/predict.py --model klebsiella_kleborate22 --present AGly_acquired,Bla_ESBL_acquired
    python scripts/predict.py --model ecoli_nit --present background__hda --top-k 10
    python scripts/predict.py --model tuberculosis_genes19 --state 1000000000000000000 --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from inference import list_models, load_model  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="\n".join(__doc__.splitlines()[2:]))
    ap.add_argument("--model", help="checkpoint name (see --list-models) or path to a model.pt")
    ap.add_argument("--present", default="", help="comma-separated features already present")
    ap.add_argument("--state", default=None, help="alternative to --present: 0/1 string in feature order")
    ap.add_argument("--top-k", type=int, default=10, help="number of candidates to show (0 = all)")
    ap.add_argument("--include-non-candidates", action="store_true",
                    help="E. coli: also score genetic-background markers (manuscript CV definition)")
    ap.add_argument("--json", action="store_true", help="print JSON instead of a table")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--list-features", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    if args.list_models:
        for name, rec in list_models().items():
            print(f"{name:42s} {rec['n_features']:5d} features  {rec['description']}")
        return 0
    if not args.model:
        ap.error("--model is required (or use --list-models)")
    if args.state is not None and args.present:
        ap.error("use either --present or --state, not both")

    try:
        model = load_model(args.model, device=args.device)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    if args.list_features:
        for i, f in enumerate(model.feature_names):
            tag = "  (genetic background; not a candidate)" if model.non_candidates[i] else ""
            print(f"{i:5d}  {f}{tag}")
        return 0

    present = [x.strip() for x in args.present.split(",") if x.strip()]
    try:
        state = model.to_state(args.state, None if args.state is not None else present)
        ranking = model.rank(state, top_k=None if args.top_k <= 0 else args.top_k,
                             include_non_candidates=args.include_non_candidates)
    except (KeyError, ValueError) as exc:
        print(f"error: {exc.args[0] if exc.args else exc}", file=sys.stderr)
        return 2
    shown_present = [model.feature_names[i] for i in range(model.n_features) if state[i]]

    if args.json:
        print(json.dumps({"model": args.model, "present": shown_present, "ranking": ranking}, indent=2))
        return 0
    print(f"model   : {args.model}  ({model.release.get('description', '')})")
    print(f"present : {', '.join(shown_present) if shown_present else '(none)'}")
    if model.non_candidates.any() and not args.include_non_candidates:
        print("note    : genetic-background markers are excluded from the candidates")
    if not ranking:
        print("no eligible candidate features (all features present)")
        return 0
    print(f"\n{'rank':>4}  {'probability':>11}  feature")
    for r in ranking:
        print(f"{r['rank']:>4}  {r['probability']:>11.4f}  {r['feature']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
