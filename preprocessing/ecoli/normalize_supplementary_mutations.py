#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把 Supplementary Data 1 / ENA 派生突变表标准化为统一长表。

目标输出 schema（最小契约）:
  sample_id, trajectory_id, timepoint, time_order, feature_id,
  event_type, presence, background, antibiotic, replicate,
  gene, variant_label, source_file

设计思路:
- 兼容 snapshot 风格（某时间点观测到哪些突变存在）和 event 风格（该时间点新增/丢失哪些突变）；
- 尽量通过 CLI 传列名映射，避免把脚本写死在某一版补充表结构上；
- 产出的长表是 Route B 的唯一入口，后续 monotonic / full_temporal 编码都从这里生成。
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULTS = {
    "sample_id": "",
    "trajectory_id": "",
    "background": "",
    "antibiotic": "",
    "replicate": "",
    "timepoint": "",
    "time_order": "",
    "gene": "",
    "variant_label": "",
    "event_type": "",
    "presence": "",
}

LOSS_LABELS = {"loss", "lost", "drop", "reversion", "revert", "absent", "delete"}
GAIN_LABELS = {"gain", "gained", "appear", "appeared", "present", "acquired", "mutated"}


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_output() -> Path:
    return repo_root() / "data" / "ecoli2026" / "derived" / "genotype_route_b" / "mutations_long.csv"


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xlsm", ".xls"}:
        return pd.read_excel(path)
    best_df = None
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python")
            if best_df is None or df.shape[1] > best_df.shape[1]:
                best_df = df
        except Exception:
            continue
    if best_df is None:
        raise SystemExit(f"ERROR: 无法读取输入文件: {path}")
    return best_df


def get_series(df: pd.DataFrame, col: str | None, default: str = "") -> pd.Series:
    if col and col in df.columns:
        return df[col].fillna("").astype(str).str.strip()
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def first_non_empty(parts: Iterable[str], fallback: str) -> str:
    vals = [str(p).strip() for p in parts if str(p).strip()]
    return "__".join(vals) if vals else fallback


def parse_presence(raw: str) -> int:
    txt = str(raw).strip().lower()
    if txt in {"", "nan", "none"}:
        return 1
    if txt in {"1", "true", "yes", "present", "mutated", "gain", "gained"}:
        return 1
    if txt in {"0", "false", "no", "absent", "loss", "lost", "reversion", "revert"}:
        return 0
    try:
        return 1 if float(txt) > 0 else 0
    except Exception:
        return 1


def normalize_event_type(raw: str, presence: int) -> str:
    txt = str(raw).strip().lower()
    if txt in LOSS_LABELS:
        return "loss"
    if txt in GAIN_LABELS:
        return "gain"
    return "gain" if presence == 1 else "loss"


def infer_time_order(df: pd.DataFrame, trajectory_col: str, timepoint_col: str, explicit_col: str | None) -> pd.Series:
    if explicit_col and explicit_col in df.columns:
        vals = pd.to_numeric(df[explicit_col], errors="coerce")
        if vals.notna().all():
            return vals.astype(int).astype(str)

    as_num = pd.to_numeric(df[timepoint_col], errors="coerce")
    if as_num.notna().all():
        return as_num.astype(int).astype(str)

    ranked = (
        df.assign(_ord=df.groupby(trajectory_col, sort=False).cumcount() + 1)["_ord"]
        .astype(int)
        .astype(str)
    )
    return ranked


def main() -> int:
    ap = argparse.ArgumentParser(description="Normalize Supplementary Data 1 / ENA-derived mutation table")
    ap.add_argument("--input", required=True, type=Path, help="原始突变表 (csv/tsv/xlsx)")
    ap.add_argument("--output", type=Path, default=default_output(), help="标准化长表输出路径")
    ap.add_argument("--input-kind", choices=["snapshot", "event"], default="snapshot",
                    help="snapshot=每行表示该 sample 中存在的突变; event=每行表示时间点发生的 gain/loss")
    ap.add_argument("--sample-col", default=None)
    ap.add_argument("--trajectory-col", default=None)
    ap.add_argument("--background-col", default=None)
    ap.add_argument("--antibiotic-col", default=None)
    ap.add_argument("--replicate-col", default=None)
    ap.add_argument("--time-col", default=None, help="时间标签列；若缺失则退化为 sample 顺序")
    ap.add_argument("--time-order-col", default=None, help="显式数值时间顺序列")
    ap.add_argument("--feature-col", default=None, help="优先使用的特征列；可直接是 mutation_id/gene_variant")
    ap.add_argument("--gene-col", default=None)
    ap.add_argument("--variant-col", default=None)
    ap.add_argument("--event-col", default=None, help="event 风格输入时指定 gain/loss 列")
    ap.add_argument("--presence-col", default=None, help="snapshot 风格输入时可选的 presence 列")
    args = ap.parse_args()

    raw = read_table(args.input)
    raw.columns = [str(c).strip() for c in raw.columns]
    raw = raw.fillna("")

    if args.feature_col and args.feature_col not in raw.columns:
        raise SystemExit(f"ERROR: 找不到 feature 列: {args.feature_col}")
    if args.gene_col and args.gene_col not in raw.columns:
        raise SystemExit(f"ERROR: 找不到 gene 列: {args.gene_col}")
    if args.variant_col and args.variant_col not in raw.columns:
        raise SystemExit(f"ERROR: 找不到 variant 列: {args.variant_col}")

    background = get_series(raw, args.background_col)
    antibiotic = get_series(raw, args.antibiotic_col)
    replicate = get_series(raw, args.replicate_col)

    if args.trajectory_col and args.trajectory_col in raw.columns:
        trajectory_id = get_series(raw, args.trajectory_col)
    else:
        trajectory_id = pd.Series(
            [first_non_empty([background.iloc[i], antibiotic.iloc[i], replicate.iloc[i]], f"trajectory_{i+1:05d}")
             for i in range(len(raw))],
            index=raw.index,
            dtype="object",
        )

    if args.time_col and args.time_col in raw.columns:
        timepoint = get_series(raw, args.time_col)
    else:
        timepoint = pd.Series([str(i + 1) for i in range(len(raw))], index=raw.index, dtype="object")

    temp = pd.DataFrame({
        "trajectory_id": trajectory_id,
        "timepoint": timepoint,
    })
    time_order = infer_time_order(temp, "trajectory_id", "timepoint", args.time_order_col)

    if args.sample_col and args.sample_col in raw.columns:
        sample_id = get_series(raw, args.sample_col)
    else:
        sample_id = pd.Series(
            [first_non_empty([trajectory_id.iloc[i], f"t{time_order.iloc[i]}", timepoint.iloc[i]], f"sample_{i+1:05d}")
             for i in range(len(raw))],
            index=raw.index,
            dtype="object",
        )

    gene = get_series(raw, args.gene_col)
    variant_label = get_series(raw, args.variant_col)
    if args.feature_col:
        feature_id = get_series(raw, args.feature_col)
    else:
        feature_id = pd.Series(
            [first_non_empty([gene.iloc[i], variant_label.iloc[i]], f"feature_{i+1:05d}")
             for i in range(len(raw))],
            index=raw.index,
            dtype="object",
        )

    if feature_id.eq("").any():
        raise SystemExit("ERROR: 标准化后存在空 feature_id，请检查 feature/gene/variant 列映射")

    if args.input_kind == "snapshot":
        presence = get_series(raw, args.presence_col, default="1").map(parse_presence)
        event_type = pd.Series(["snapshot"] * len(raw), index=raw.index, dtype="object")
    else:
        if not args.event_col or args.event_col not in raw.columns:
            raise SystemExit("ERROR: input-kind=event 时必须提供合法的 --event-col")
        raw_event = get_series(raw, args.event_col)
        presence = raw_event.map(lambda x: 0 if str(x).strip().lower() in LOSS_LABELS else 1)
        event_type = pd.Series(
            [normalize_event_type(raw_event.iloc[i], int(presence.iloc[i])) for i in range(len(raw))],
            index=raw.index,
            dtype="object",
        )

    out = pd.DataFrame({
        "sample_id": sample_id.astype(str),
        "trajectory_id": trajectory_id.astype(str),
        "background": background.astype(str),
        "antibiotic": antibiotic.astype(str),
        "replicate": replicate.astype(str),
        "timepoint": timepoint.astype(str),
        "time_order": pd.to_numeric(time_order, errors="coerce").fillna(0).astype(int),
        "feature_id": feature_id.astype(str),
        "gene": gene.astype(str),
        "variant_label": variant_label.astype(str),
        "event_type": event_type.astype(str),
        "presence": pd.to_numeric(presence, errors="coerce").fillna(0).astype(int),
        "source_file": str(args.input.resolve()),
    })

    out = out[out["sample_id"].astype(str).str.strip() != ""].copy()
    out = out[out["trajectory_id"].astype(str).str.strip() != ""].copy()
    out = out[out["feature_id"].astype(str).str.strip() != ""].copy()
    out = out.sort_values(["trajectory_id", "time_order", "sample_id", "feature_id"]).reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    n_samples = out["sample_id"].nunique()
    n_traj = out["trajectory_id"].nunique()
    n_feats = out["feature_id"].nunique()
    print(
        "[ecoli] normalized mutations rows=%d samples=%d trajectories=%d features=%d -> %s"
        % (len(out), n_samples, n_traj, n_feats, args.output)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
