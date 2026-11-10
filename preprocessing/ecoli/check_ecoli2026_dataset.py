#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""审计 `ecoli2026` 数据是否适合 EvoAMR / HyperTraPS 分析。

这个脚本会：
1. 扫描 `fsepi_data` 下各实验目录；
2. 统计 plate layout、growth rate、OD 曲线文件；
3. 提取 plate-level strain/well manifest；
4. 检查是否存在可直接用于 HyperTraPS/当前 pipeline 的基因型、barcode、lineage、transition 文件；
5. 输出 JSON 结论，帮助判断下一步该做 phenotype 分析还是数据转换。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from openpyxl import load_workbook
except Exception:  # pragma: no cover - optional dependency
    load_workbook = None

PLATE_RE = re.compile(r"plate(\d+)", re.IGNORECASE)
LONG_DIGIT_RE = re.compile(r"(\d{4,})")
PLATE_LAYOUT_RE = re.compile(r"plate\d+_(\d{4,})", re.IGNORECASE)
WELL_RE = re.compile(r"([A-H](?:1[0-2]|[1-9]))")
GENOMIC_HINTS = (
    "variant",
    "mutation",
    "mutations",
    "genome",
    "sequence",
    "sequencing",
    "snv",
    "snp",
    "vcf",
    "fasta",
    "fastq",
)
PIPELINE_HINTS = (
    "barcode",
    "transition",
    "lineage",
    "phylo",
    "newick",
    "tree",
    "hiercc",
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_dir() -> Path:
    return repo_root() / "data" / "ecoli2026"


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    last_error = None
    best_df = None
    for sep in [",", "\t", ";"]:
        try:
            df = pd.read_csv(path, sep=sep, engine="python", **kwargs)
            if best_df is None or df.shape[1] > best_df.shape[1]:
                best_df = df
        except Exception as exc:  # pragma: no cover - best-effort sniffing
            last_error = exc
    if best_df is None:
        raise RuntimeError(f"无法读取文件: {path} ({last_error})")
    return best_df


def extract_plate_id(path: Path) -> str:
    layout_match = PLATE_LAYOUT_RE.search(path.name)
    if layout_match:
        return layout_match.group(1)
    long_digits = LONG_DIGIT_RE.findall(path.name)
    if long_digits:
        return long_digits[0]
    match = PLATE_RE.search(path.name)
    return match.group(1) if match else path.stem


def extract_wells_from_columns(columns: Iterable[str]) -> list[str]:
    wells = []
    for col in columns:
        match = WELL_RE.search(str(col))
        if match:
            wells.append(match.group(1))
    return sorted(set(wells))


def parse_layout_file(path: Path, experiment: str) -> list[dict]:
    df = read_table(path, dtype=str, keep_default_na=False)
    if df.empty or df.shape[1] < 2:
        return []

    row_col = df.columns[0]
    column_headers = [str(c).strip() for c in df.columns[1:]]
    records = []
    for _, row in df.iterrows():
        row_label = str(row[row_col]).strip()
        if not row_label or row_label.lower() == "nan":
            continue
        for col_header in column_headers:
            value = str(row[col_header]).strip()
            if not value or value.lower() == "nan":
                continue
            well = f"{row_label}{col_header}"
            records.append(
                {
                    "experiment": experiment,
                    "plate_id": extract_plate_id(path),
                    "well": well,
                    "strain_label": value,
                    "is_lb_control": value.upper() == "LB",
                    "layout_file": path.name,
                }
            )
    return records


def inspect_experiment(exp_dir: Path) -> tuple[dict, list[dict]]:
    data_dir = exp_dir / "data"
    layout_dir = exp_dir / "TF_plate_layouts_strain_names"
    growth_files = sorted(data_dir.glob("growth_rates*.csv")) if data_dir.exists() else []
    od_files = sorted(data_dir.glob("ODdata_bg_sub_with_exptime*.csv")) if data_dir.exists() else []
    mec_files = sorted(data_dir.glob("MEC*.csv")) if data_dir.exists() else []
    layout_files = sorted(layout_dir.glob("*.csv")) if layout_dir.exists() else []

    manifest_rows = []
    for layout_file in layout_files:
        manifest_rows.extend(parse_layout_file(layout_file, exp_dir.name))

    growth_plate_ids = {extract_plate_id(p) for p in growth_files}
    od_plate_ids = {extract_plate_id(p) for p in od_files}
    layout_plate_ids = {extract_plate_id(p) for p in layout_files}

    unique_strains = sorted({r["strain_label"] for r in manifest_rows if not r["is_lb_control"]})
    lb_controls = sum(1 for r in manifest_rows if r["is_lb_control"])

    example_growth = None
    growth_well_counts = {}
    if growth_files:
        growth_df = read_table(growth_files[0], dtype=str, keep_default_na=False, nrows=5)
        wells = extract_wells_from_columns(growth_df.columns)
        growth_well_counts[growth_files[0].name] = len(wells)
        example_growth = {
            "file": growth_files[0].name,
            "n_columns": int(growth_df.shape[1]),
            "example_columns": [str(c) for c in list(growth_df.columns[:12])],
            "n_detected_wells": len(wells),
        }

    example_od = None
    if od_files:
        od_df = read_table(od_files[0], dtype=str, keep_default_na=False, nrows=3)
        wells = extract_wells_from_columns(od_df.columns)
        example_od = {
            "file": od_files[0].name,
            "n_columns": int(od_df.shape[1]),
            "example_columns": [str(c) for c in list(od_df.columns[:12])],
            "n_detected_wells": len(wells),
        }

    summary = {
        "experiment": exp_dir.name,
        "n_growth_files": len(growth_files),
        "n_od_files": len(od_files),
        "n_mec_files": len(mec_files),
        "n_layout_files": len(layout_files),
        "n_manifest_rows": len(manifest_rows),
        "n_unique_non_control_labels": len(unique_strains),
        "n_lb_controls": lb_controls,
        "plate_id_overlap": {
            "layout": sorted(layout_plate_ids),
            "growth": sorted(growth_plate_ids),
            "od": sorted(od_plate_ids),
            "growth_subset_of_layout": growth_plate_ids.issubset(layout_plate_ids) if layout_plate_ids else False,
            "od_subset_of_layout": od_plate_ids.issubset(layout_plate_ids) if layout_plate_ids else False,
        },
        "example_growth": example_growth,
        "example_od": example_od,
        "sample_strain_labels": unique_strains[:20],
    }
    return summary, manifest_rows


def detect_analysis_readiness(dataset_dir: Path) -> dict:
    all_files = [p for p in dataset_dir.rglob("*") if p.is_file()]
    lower_names = [p.name.lower() for p in all_files]
    genomic_files = sorted(
        str(p.relative_to(dataset_dir))
        for p in all_files
        if any(token in p.name.lower() for token in GENOMIC_HINTS)
        or p.suffix.lower() in {".vcf", ".fasta", ".fa", ".fastq", ".fq"}
    )
    pipeline_ready_files = sorted(
        str(p.relative_to(dataset_dir))
        for p in all_files
        if any(token in p.name.lower() for token in PIPELINE_HINTS)
    )

    has_barcode = any("barcode" in name for name in lower_names)
    has_transitions = any("transition" in name for name in lower_names)
    has_lineage = any(token in name for token in ("lineage", "phylo", "tree", "hiercc") for name in lower_names)
    has_genotype = len(genomic_files) > 0

    direct_hypertraps_ready = has_barcode or has_transitions
    direct_pipeline_ready = has_barcode and has_lineage

    reasons = []
    if not has_genotype:
        reasons.append("未发现突变/序列/VCF 等基因型层文件")
    if not has_barcode:
        reasons.append("未发现可直接输入 HyperTraPS 的 0/1 barcode/state matrix")
    if not has_lineage:
        reasons.append("未发现祖先-后代、树或谱系分组信息")
    if not has_transitions:
        reasons.append("未发现现成 transitions 文件")

    return {
        "direct_hypertraps_ready": direct_hypertraps_ready,
        "direct_pipeline_ready": direct_pipeline_ready,
        "has_genotype_or_variant_level_files": has_genotype,
        "genomic_or_variant_file_examples": genomic_files[:20],
        "pipeline_like_file_examples": pipeline_ready_files[:20],
        "blocking_reasons": reasons,
        "recommended_position": (
            "可以用于表型耐药/生长反应分析，以及后续构建派生 phenotype feature；"
            "但暂不适合直接送入 HyperTraPS 或当前 EvoAMR dataset builder。"
        ),
    }


def inspect_source_workbooks(dataset_dir: Path) -> dict:
    result = {}
    if load_workbook is None:
        return {"available": False, "reason": "openpyxl 不可用"}

    for name in ["SourceData_MainFigures.xlsx", "SourceData_SupplementaryFigures.xlsx"]:
        path = dataset_dir / name
        if not path.exists():
            continue
        wb = load_workbook(path, read_only=True, data_only=True)
        result[name] = {
            "n_sheets": len(wb.sheetnames),
            "sheet_names": wb.sheetnames[:20],
        }
    return result or {"available": False, "reason": "未发现 source data workbook"}


def build_report(dataset_dir: Path) -> dict:
    raw_dir = dataset_dir / "fsepi_data"
    experiments = []
    manifest_rows = []
    if raw_dir.exists():
        for exp_dir in sorted(p for p in raw_dir.iterdir() if p.is_dir() and not p.name.startswith(".")):
            summary, rows = inspect_experiment(exp_dir)
            experiments.append(summary)
            manifest_rows.extend(rows)

    manifest_df = pd.DataFrame(manifest_rows)
    if not manifest_df.empty:
        manifest_summary = {
            "n_rows": int(len(manifest_df)),
            "n_experiments": int(manifest_df["experiment"].nunique()),
            "n_plates": int(manifest_df[["experiment", "plate_id"]].drop_duplicates().shape[0]),
            "n_unique_labels": int(manifest_df.loc[~manifest_df["is_lb_control"], "strain_label"].nunique()),
            "n_lb_controls": int(manifest_df["is_lb_control"].sum()),
        }
    else:
        manifest_summary = {
            "n_rows": 0,
            "n_experiments": 0,
            "n_plates": 0,
            "n_unique_labels": 0,
            "n_lb_controls": 0,
        }

    return {
        "dataset_dir": str(dataset_dir),
        "source_workbooks": inspect_source_workbooks(dataset_dir),
        "experiments": experiments,
        "plate_manifest_summary": manifest_summary,
        "analysis_readiness": detect_analysis_readiness(dataset_dir),
        "next_steps": [
            "先把 plate layout + growth/OD 文件整理为统一长表（experiment, plate_id, well, strain_label, dilution, metric）。",
            "若目标是 HyperTraPS，需要额外构建离散状态/事件：例如从多药条件下的 phenotype class 或函数模块缺失标签派生 0/1 特征。",
            "若目标是当前 EvoAMR pipeline，需要补齐 sample-level barcode 和 lineage/tree/HierCC 等结构信息，或新增专门适配 phenotype trajectory 的 adapter。",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit ecoli2026 dataset readiness")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir(),
        help="`ecoli2026` 根目录，默认指向仓库内 `evoamr/data/ecoli2026`",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="可选 JSON 输出路径",
    )
    args = parser.parse_args()

    report = build_report(args.dataset_dir.resolve())
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
