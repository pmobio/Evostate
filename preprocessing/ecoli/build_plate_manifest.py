#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 `ecoli2026/fsepi_data` 的 plate layout 文件提取长表 manifest。"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

PLATE_RE = re.compile(r"plate(\d+)", re.IGNORECASE)
LONG_DIGIT_RE = re.compile(r"(\d{4,})")
PLATE_LAYOUT_RE = re.compile(r"plate\d+_(\d{4,})", re.IGNORECASE)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_raw_dir() -> Path:
    return repo_root() / "data" / "ecoli2026" / "fsepi_data"


def read_table(path: Path) -> pd.DataFrame:
    best_df = None
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", dtype=str, keep_default_na=False)
            if best_df is None or df.shape[1] > best_df.shape[1]:
                best_df = df
        except Exception:
            continue
    if best_df is None:
        raise RuntimeError(f"无法读取布局文件: {path}")
    return best_df


def plate_id_from_name(path: Path) -> str:
    layout_match = PLATE_LAYOUT_RE.search(path.name)
    if layout_match:
        return layout_match.group(1)
    long_digits = LONG_DIGIT_RE.findall(path.name)
    if long_digits:
        return long_digits[0]
    match = PLATE_RE.search(path.name)
    return match.group(1) if match else path.stem


def iter_layout_rows(layout_file: Path, experiment: str):
    df = read_table(layout_file)
    if df.empty or df.shape[1] < 2:
        return

    row_key = df.columns[0]
    col_headers = [str(c).strip() for c in df.columns[1:]]
    plate_id = plate_id_from_name(layout_file)
    for _, row in df.iterrows():
        row_label = str(row[row_key]).strip()
        if not row_label or row_label.lower() == "nan":
            continue
        for col_header in col_headers:
            strain_label = str(row[col_header]).strip()
            if not strain_label or strain_label.lower() == "nan":
                continue
            yield {
                "experiment": experiment,
                "plate_id": plate_id,
                "well": f"{row_label}{col_header}",
                "strain_label": strain_label,
                "is_lb_control": int(strain_label.upper() == "LB"),
                "layout_file": layout_file.name,
            }


def build_manifest(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for exp_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
        layout_dir = exp_dir / "TF_plate_layouts_strain_names"
        if not layout_dir.exists():
            continue
        for layout_file in sorted(layout_dir.glob("*.csv")):
            rows.extend(iter_layout_rows(layout_file, exp_dir.name))
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build plate manifest for ecoli2026")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=default_raw_dir(),
        help="`fsepi_data` 目录",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repo_root() / "data" / "ecoli2026" / "derived" / "plate_manifest.csv",
        help="输出 CSV 路径",
    )
    args = parser.parse_args()

    manifest = build_manifest(args.raw_dir.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.output, index=False)

    print(f"[ecoli] manifest rows={len(manifest)} output={args.output}")
    if not manifest.empty:
        print(
            "[ecoli] experiments=%d plates=%d unique_labels=%d lb_controls=%d"
            % (
                manifest["experiment"].nunique(),
                manifest[["experiment", "plate_id"]].drop_duplicates().shape[0],
                manifest.loc[manifest["is_lb_control"] == 0, "strain_label"].nunique(),
                int(manifest["is_lb_control"].sum()),
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
