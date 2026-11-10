#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 `SourceData_MainFigures.xlsx` / `SourceData_SupplementaryFigures.xlsx` 的真实突变矩阵提取终点基因型长表。

当前实现针对 Supplementary Data 1 对应 sheet：
- `Fig3b_MEC`   (97 行)
- `FigS8_NIT`   (106 行)
- `FigS8_TMP`   (108 行)

输出：
- `mutations_long.csv`：标准化长表入口
- `sample_manifest.csv`：按 sample 汇总的元数据
- `source_matrix_manifest.json`：提取统计与假设记录

数据结构（已用 `fsepi_data` 的板布局与运行日志核实）：
- **本表没有时间维度。** 每一行 = 一个演化终点 population 的固定突变谱。
  证据: (1) `CurrentExperiment_info/Plates_info_*.txt` 显示 9 块板在同一天 3 小时内
  全部启动、各占一个培养箱位并行运行约 20 天, 因此 plate_id 不是时间点;
  (2) 同一实验内连号三块板 (如 17221/17222/17223) 的 `plate_layouts/*.csv` 字节级相同,
  同一孔位是同一菌株, 即 3 个平行 replicate (论文: 258 株 × 3 replicates);
  (3) TMP 全部 108 行的 `sample_tag` 字面为 `end`;
  (4) 论文 Fig 3b 图注为 "mutations fixed in all sequenced **evolved** populations"。
- 每行 `title` 为 `plate_id + endpoint_tag + well + strain_label` 四段式；
  `endpoint_tag` 是取样测序时对应的 morbidostat 读数文件编号 (终点时刻标记),
  可在 `data/ABconc_with_exptime_plate<plate_id>.csv` 的 `original_file` 列中找到。
- `strain_group_id = antibiotic + well + strain_label` 把同一菌株的 3 个 replicate
  聚为一组, 这是 grouped CV 与 parallelism index 检验的正确分组单元；
- `replicate_index` 为该 population 在其 strain group 内按 plate_id 升序的序号 (1..3)，
  **不是时间顺序**；
- `trajectory_id` 被刻意设为等于 `population_id`（每条长度为 1），以避免下游把
  replicate 之间的差异误当作演化转移。真实时间轴在表型数据里，见
  `build_phenotype_trajectories.py`。
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

SHEET_CONFIGS = [
    {
        "file": "SourceData_MainFigures.xlsx",
        "sheet": "Fig3b_MEC",
        "antibiotic": "MEC",
    },
    {
        "file": "SourceData_SupplementaryFigures.xlsx",
        "sheet": "FigS8_NIT",
        "antibiotic": "NIT",
    },
    {
        "file": "SourceData_SupplementaryFigures.xlsx",
        "sheet": "FigS8_TMP",
        "antibiotic": "TMP",
    },
]

SAMPLE_ID_RE = re.compile(
    r"^(?P<plate_id>[^_]+)_(?P<sample_tag>[^_]+)_(?P<well>[A-H](?:1[0-2]|[1-9]))_(?P<strain_label>.+)$"
)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_dir() -> Path:
    return repo_root() / "data" / "ecoli2026"


def default_outdir() -> Path:
    return default_dataset_dir() / "derived" / "genotype_route_b"


def clean_feature_name(name: object) -> str:
    text = str(name).strip()
    if not text or text.lower() == "nan":
        raise SystemExit("ERROR: 突变矩阵中存在空特征列名")
    return text


def load_matrix(path: Path, sheet_name: str) -> pd.DataFrame:
    df = pd.read_excel(path, sheet_name=sheet_name)
    df.columns = [str(c).strip() for c in df.columns]
    if df.empty:
        raise SystemExit(f"ERROR: 工作表为空: {path.name}#{sheet_name}")
    first_col = df.columns[0]
    if first_col != "title":
        df = df.rename(columns={first_col: "title"})
    df = df[df["title"].notna()].copy()
    df["title"] = df["title"].astype(str).str.strip()
    df = df[df["title"] != ""].reset_index(drop=True)
    if df.empty:
        raise SystemExit(f"ERROR: 工作表缺少有效 sample 标识: {path.name}#{sheet_name}")

    feature_df = df.drop(columns=["title"]).copy()
    feature_df.columns = [clean_feature_name(c) for c in feature_df.columns]
    feature_df = feature_df.T.groupby(level=0).max().T
    feature_df = feature_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    feature_df = (feature_df > 0).astype(int)
    out = pd.concat([df[["title"]], feature_df], axis=1)
    return out


def parse_sample_ids(sample_ids: pd.Series, antibiotic: str, source_file: str, source_sheet: str) -> pd.DataFrame:
    """把 `title` 解析为终点 population 级元数据 (无时间维度)。"""
    rows: list[dict] = []
    for sample_id in sample_ids.astype(str):
        match = SAMPLE_ID_RE.match(sample_id)
        if not match:
            raise SystemExit(f"ERROR: 无法解析 sample_id: {sample_id}")
        plate_id = match.group("plate_id")
        well = match.group("well")
        strain_label = match.group("strain_label")
        strain_group_label = f"{well}_{strain_label}"
        rows.append(
            {
                "sample_id": sample_id,
                "population_id": sample_id,
                # trajectory_id 等于 population_id: 每条长度为 1, 下游不会构造出伪转移。
                "trajectory_id": sample_id,
                "strain_group_id": f"{antibiotic}:{strain_group_label}",
                "strain_group_label": strain_group_label,
                "background": strain_label,
                "antibiotic": antibiotic,
                "plate_id": plate_id,
                "well": well,
                "strain_label": strain_label,
                # endpoint_tag: 取样测序时对应的 morbidostat 读数文件编号 (终点时刻标记)。
                "endpoint_tag": match.group("sample_tag"),
                "sample_tag": match.group("sample_tag"),
                "event_type": "endpoint",
                # 保留常量 time_order 仅为下游 schema 兼容; 本数据无时间轴。
                "time_order": 1,
                "source_file": source_file,
                "source_sheet": source_sheet,
            }
        )
    out = pd.DataFrame(rows)
    # replicate_index: strain group 内按 plate_id 数值升序编号, 表示第几个平行 replicate。
    out["_plate_num"] = pd.to_numeric(out["plate_id"], errors="coerce")
    out["replicate_index"] = (
        out.sort_values(["strain_group_id", "_plate_num", "sample_id"])
        .groupby("strain_group_id")
        .cumcount()
        + 1
    )
    out = out.drop(columns=["_plate_num"])
    out["replicate"] = out["replicate_index"]
    return out


def build_sheet_long(dataset_dir: Path, cfg: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    path = dataset_dir / cfg["file"]
    if not path.exists():
        raise SystemExit(f"ERROR: 找不到源文件: {path}")

    matrix_df = load_matrix(path, cfg["sheet"])
    sample_meta = parse_sample_ids(
        matrix_df["title"],
        antibiotic=cfg["antibiotic"],
        source_file=cfg["file"],
        source_sheet=cfg["sheet"],
    )

    feature_matrix = matrix_df.drop(columns=["title"]).copy()
    feature_matrix.index = sample_meta["sample_id"]
    feature_matrix.index.name = "sample_id"

    long_df = (
        feature_matrix.stack()
        .rename("presence")
        .reset_index()
        .rename(columns={"level_1": "feature_id"})
    )
    long_df = long_df[long_df["presence"] > 0].copy()
    long_df = long_df.merge(sample_meta, on="sample_id", how="left", validate="many_to_one")
    long_df["gene"] = long_df["feature_id"].astype(str)
    long_df["variant_label"] = "present_in_matrix"

    sample_summary = sample_meta.copy()
    n_positive = feature_matrix.sum(axis=1).astype(int)
    sample_summary["n_positive_features"] = sample_summary["sample_id"].map(n_positive.to_dict()).fillna(0).astype(int)

    traj_lengths = sample_summary.groupby("strain_group_id")["sample_id"].nunique()
    plate_ids = sorted(sample_summary["plate_id"].unique().tolist(), key=lambda x: int(str(x)))
    sheet_manifest = {
        "file": cfg["file"],
        "sheet": cfg["sheet"],
        "antibiotic": cfg["antibiotic"],
        "has_temporal_axis": False,
        "n_samples": int(sample_summary["sample_id"].nunique()),
        "n_populations": int(sample_summary["population_id"].nunique()),
        "n_strain_groups": int(sample_summary["strain_group_id"].nunique()),
        "n_backgrounds": int(sample_summary["background"].nunique()),
        "n_features": int(feature_matrix.shape[1]),
        "n_positive_rows": int(len(long_df)),
        "plate_ids": plate_ids,
        "replicates_per_strain_group_distribution": {
            str(int(k)): int(v)
            for k, v in traj_lengths.value_counts().sort_index().items()
        },
        "max_positive_features_per_sample": int(sample_summary["n_positive_features"].max()),
        "median_positive_features_per_sample": float(sample_summary["n_positive_features"].median()),
    }
    return long_df, sample_summary, sheet_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract Route B long table from supplementary mutation matrices")
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=default_dataset_dir(),
        help="`ecoli2026` 数据目录",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=default_outdir(),
        help="输出目录，默认 `derived/genotype_route_b`",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    long_parts: list[pd.DataFrame] = []
    sample_parts: list[pd.DataFrame] = []
    manifests: list[dict] = []
    for cfg in SHEET_CONFIGS:
        long_df, sample_df, manifest = build_sheet_long(dataset_dir, cfg)
        long_parts.append(long_df)
        sample_parts.append(sample_df)
        manifests.append(manifest)

    mutations_long = pd.concat(long_parts, ignore_index=True)
    mutations_long = mutations_long.sort_values(
        ["antibiotic", "strain_group_id", "replicate_index", "sample_id", "feature_id"]
    ).reset_index(drop=True)

    sample_manifest = pd.concat(sample_parts, ignore_index=True)
    sample_manifest = sample_manifest.drop_duplicates(subset=["sample_id"]).sort_values(
        ["antibiotic", "strain_group_id", "replicate_index", "sample_id"]
    ).reset_index(drop=True)

    long_path = outdir / "mutations_long.csv"
    sample_path = outdir / "sample_manifest.csv"
    manifest_path = outdir / "source_matrix_manifest.json"

    mutations_long.to_csv(long_path, index=False)
    sample_manifest.to_csv(sample_path, index=False)

    payload = {
        "source": "Supplementary Data 1 matrices from SourceData_MainFigures.xlsx / SourceData_SupplementaryFigures.xlsx",
        "has_temporal_axis": False,
        "unit_of_observation": "evolved population endpoint (one row per sequenced population)",
        "assumptions": [
            "title = plate_id + endpoint_tag + well + strain_label",
            "population_id = sample_id; trajectory_id = population_id (length 1, no transitions)",
            "strain_group_id = antibiotic + well + strain_label groups the 3 parallel replicates",
            "replicate_index = rank of plate_id within a strain group (NOT time order)",
            "endpoint_tag = morbidostat readout file id at sampling time (all samples are endpoints)",
            "only positive matrix entries are written to mutations_long.csv; absence is recovered downstream as 0",
        ],
        "verified_against": [
            "fsepi_data/*/CurrentExperiment_info/Plates_info_*.txt (all 9 plates started same day, run in parallel)",
            "fsepi_data/*/plate_layouts/plate{1,2,3}_*_layout_*.csv (byte-identical -> same well = same strain)",
            "fsepi_data/strain_names_plate_GPm{1,2,3}.csv (87 strains each, disjoint, union = 258)",
        ],
        "temporal_data_location": "fsepi_data/*/data/ABconc_with_exptime_plate*.csv (see build_phenotype_trajectories.py)",
        "sheets": manifests,
        "global": {
            "n_samples": int(sample_manifest["sample_id"].nunique()),
            "n_populations": int(sample_manifest["population_id"].nunique()),
            "n_strain_groups": int(sample_manifest["strain_group_id"].nunique()),
            "n_backgrounds": int(sample_manifest["background"].nunique()),
            "n_features_total": int(mutations_long["feature_id"].nunique()),
            "n_rows": int(len(mutations_long)),
            "antibiotics": sorted(sample_manifest["antibiotic"].unique().tolist()),
        },
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        "[ecoli] extracted endpoint matrices (no temporal axis): rows=%d populations=%d strain_groups=%d features=%d -> %s"
        % (
            len(mutations_long),
            sample_manifest["population_id"].nunique(),
            sample_manifest["strain_group_id"].nunique(),
            mutations_long["feature_id"].nunique(),
            long_path,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
