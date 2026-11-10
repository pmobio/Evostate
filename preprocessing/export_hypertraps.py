#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Export dataset to HyperTraPS transitions.txt (paired source/target lines)."""
import argparse, csv, json, os

def from_evam(path):
    rows = list(csv.reader(open(path)))
    hdr = rows[0]; n = len(hdr); pairs = []
    for r in rows[1:]:
        st = [x.strip() for x in r]
        if len(st) != n: continue
        if sum(int(x) for x in st) == 0: continue
        pairs.append((["0"]*n, st))
    return pairs, hdr

def _feature_labels(transitions_path, n):
    """真实特征名: 优先读同目录/父目录的 transitions_summary.json 或 manifest.json
    的 feature_columns (由 build_transitions/stratify_transitions 写出); 找不到才退化为 f0..fn。"""
    d = os.path.dirname(os.path.abspath(transitions_path))
    cands = [os.path.join(d, "transitions_summary.json"),
             os.path.join(os.path.dirname(d), "manifest.json"),
             os.path.join(d, "manifest.json")]
    for c in cands:
        if os.path.exists(c):
            try:
                cols = json.load(open(c)).get("feature_columns")
                if cols and len(cols) == n:
                    return list(cols)
            except Exception:
                pass
    return ["f%d" % i for i in range(n)]

def _ensure_irreversible_input(path):
    """HyperTraPS 对齐口径: 这里只接受不可逆/gain-only transition 表。
    build_transitions 现在会并行产出 selective_reversible 版本, 但那份不应直接喂给 HyperTraPS。"""
    rows = list(csv.DictReader(open(path)))
    if not rows:
        return rows
    bad_strategy = sorted({str(r.get("transition_strategy", "")).strip()
                           for r in rows
                           if str(r.get("transition_strategy", "")).strip()
                           and str(r.get("transition_strategy", "")).strip() != "irreversible"})
    n_loss = 0
    for r in rows:
        raw = str(r.get("n_lost", "0")).strip() or "0"
        try:
            n_loss += int(raw)
        except ValueError:
            pass
    if bad_strategy or n_loss > 0:
        raise SystemExit(
            "export_hypertraps.py 只接受不可逆/gain-only transitions。"
            f" 检测到 strategy={bad_strategy or ['irreversible?']} 且总 n_lost={n_loss}; "
            "请改用 build_transitions 产出的 transitions.csv 或 transitions_irreversible.csv，"
            "不要直接使用 transitions_selective_reversible.csv。")
    return rows


def from_transitions(path):
    rows = _ensure_irreversible_input(path)
    pairs = []; n = 0
    for r in rows:
        src = list(r["src_state"].strip())
        dst = list(r["dest_state"].strip())
        n = len(dst); pairs.append((src, dst))
    return pairs, _feature_labels(path, n)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evam", default=None)
    ap.add_argument("--transitions", default=None)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args(); os.makedirs(args.outdir, exist_ok=True)
    if args.evam:
        pairs, hdr = from_evam(args.evam); mode = "cross-sectional"
    elif args.transitions:
        pairs, hdr = from_transitions(args.transitions); mode = "phylogenetic"
    else:
        raise SystemExit("need --evam or --transitions")
    tpath = os.path.join(args.outdir, "transitions.txt")
    with open(tpath, "w") as f:
        for s, d in pairs:
            f.write(" ".join(s) + chr(10)); f.write(" ".join(d) + chr(10))
    with open(os.path.join(args.outdir, "labels.csv"), "w") as f:
        for h in hdr: f.write(str(h) + chr(10))
    print("[export] mode=%s transitions=%d lines=%d L=%d" % (mode, len(pairs), len(pairs)*2, len(hdr)))

if __name__ == "__main__":
    main()
