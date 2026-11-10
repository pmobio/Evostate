#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 `fsepi_data` 构建 ecoli2026 的真实表型演化时间序列。

背景
----
`SourceData_*.xlsx` 的突变矩阵只有演化终点 (每行一个 population, 无时间维度)。
真正的时间轴在 morbidostat 的运行数据里: 每块 96 孔板每 ~4.2 h 一次稀释循环,
每次循环记录该孔的抗生素浓度 (= 该 population 当时的 IC50) 与生长速率, 持续约 20 天。

输入
----
`fsepi_data/<experiment>/`
  data/ABconc_with_exptime_plate<plate_id>.csv    tab 分隔, 行=稀释循环, 列=96 孔 + 时间列
  data/growth_rates_*_plate<plate_id>.csv         逗号分隔, 列名 `<well>_gr_fit`
  plate_layouts/plate<n>_<plate_id>_layout_*.csv  8x12 网格, 值为 s<k> / lacA / LB
  plate_layouts/antibiotics_plate<n>_*.csv        (可选) 逐孔抗生素, 用于多药混合板
`fsepi_data/strain_names_plate_GPm{1,2,3}.csv`    s<k> -> 缺失基因 (各 87 株, 互斥, 并集 258)

板 -> GPm 的对应关系
-------------------
9 板实验中连号三块板布局字节相同, 是同一组 87 株的 3 个平行 replicate;
按 plate_id 升序, 第 1-3 块用 GPm1, 第 4-6 块用 GPm2, 第 7-9 块用 GPm3。
脚本会用基因型 manifest (若存在) 交叉校验该映射并报告命中率。

输出
----
`derived/phenotype/`
  phenotype_trajectories.csv   长表: 每 (population, 时间点) 一行
  phenotype_populations.csv    每个 population 一行的汇总 (初始/终点 IC50、fold increase、AUC 等)
  manifest.json                提取统计、映射校验结果、假设记录
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

# 每个实验目录对应的固定抗生素; None 表示逐孔抗生素由 antibiotics 布局给出。
EXPERIMENTS: list[dict] = [
    {"dir": "Results_MEC_9plates_20052022", "antibiotic": "MEC", "n_plate_groups": 3},
    {"dir": "Results_NIT_9plates_21062021", "antibiotic": "NIT", "n_plate_groups": 3},
    {"dir": "Results_TMP_9plates_12082021", "antibiotic": "TMP", "n_plate_groups": 3},
    {"dir": "Results_MEC_NIT_TMP_6plates_24042023", "antibiotic": None, "n_plate_groups": None},
]

GPM_FILES = {
    1: "strain_names_plate_GPm1bkpcorrected.csv",
    2: "strain_names_plate_GPm2.csv",
    3: "strain_names_plate_GPm3.csv",
}

WELL_RE = re.compile(r"^[A-H](?:1[0-2]|[1-9])$")
BLANK_CODES = {"LB", "EMPTY", "BLANK", ""}
TIME_COL = "time[h]"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_dataset_dir() -> Path:
    return repo_root() / "data" / "ecoli2026"


def default_outdir() -> Path:
    return default_dataset_dir() / "derived" / "phenotype"


def load_gpm_maps(fsepi_dir: Path) -> dict[int, dict[str, str]]:
    """读取三份 GPm 菌株编号 -> 缺失基因映射。"""
    maps: dict[int, dict[str, str]] = {}
    for idx, fname in GPM_FILES.items():
        path = fsepi_dir / fname
        if not path.exists():
            raise SystemExit(f"ERROR: 缺少 GPm 映射文件: {path}")
        df = pd.read_csv(path)
        df = df.iloc[:, :2]
        df.columns = ["strain", "gene"]
        df["strain"] = df["strain"].astype(str).str.strip().str.strip('"')
        df["gene"] = df["gene"].astype(str).str.strip().str.strip('"')
        df = df[(df["strain"] != "") & (df["gene"] != "") & (df["gene"].str.lower() != "nan")]
        maps[idx] = dict(zip(df["strain"], df["gene"]))
    return maps


def parse_layout(path: Path) -> dict[str, str]:
    """把 8x12 布局网格解析为 well -> 编码。"""
    df = pd.read_csv(path, index_col=0)
    df.index = [str(i).strip() for i in df.index]
    df.columns = [str(c).strip() for c in df.columns]
    out: dict[str, str] = {}
    for row in df.index:
        for col in df.columns:
            well = f"{row}{col}"
            if not WELL_RE.match(well):
                continue
            value = df.loc[row, col]
            out[well] = "" if pd.isna(value) else str(value).strip()
    return out


def discover_plates(exp_dir: Path) -> pd.DataFrame:
    """从 plate_layouts 目录发现 (plate_index, plate_id, layout_path)。"""
    layout_dir = exp_dir / "plate_layouts"
    if not layout_dir.is_dir():
        raise SystemExit(f"ERROR: 缺少 plate_layouts 目录: {layout_dir}")
    rows = []
    pattern = re.compile(r"^plate(?P<idx>\d+)_(?P<plate_id>\d+)_layout")
    for path in sorted(layout_dir.glob("plate*_layout*.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        rows.append({
            "plate_index": int(match.group("idx")),
            "plate_id": match.group("plate_id"),
            "layout_path": path,
        })
    if not rows:
        raise SystemExit(f"ERROR: 未在 {layout_dir} 找到 plate 布局文件")
    df = pd.DataFrame(rows).sort_values("plate_index").reset_index(drop=True)

    ab_layouts = {}
    ab_pattern = re.compile(r"^antibiotics_plate(?P<idx>\d+)_(?P<plate_id>\d+)_layout")
    for path in sorted(layout_dir.glob("antibiotics_plate*_layout*.csv")):
        match = ab_pattern.match(path.name)
        if match:
            ab_layouts[int(match.group("idx"))] = path
    df["antibiotic_layout_path"] = df["plate_index"].map(ab_layouts)
    return df


def assign_plate_groups(plates: pd.DataFrame, n_plate_groups: int | None) -> pd.DataFrame:
    """按 plate_id 升序把板分成若干组, 每组内的板是彼此的 replicate。"""
    plates = plates.copy()
    plates["_num"] = pd.to_numeric(plates["plate_id"], errors="coerce")
    plates = plates.sort_values(["_num", "plate_index"]).reset_index(drop=True)
    n_plates = len(plates)
    if n_plate_groups and n_plates % n_plate_groups == 0:
        per_group = n_plates // n_plate_groups
        plates["plate_group"] = [i // per_group + 1 for i in range(n_plates)]
        plates["replicate_index"] = [i % per_group + 1 for i in range(n_plates)]
    else:
        # 未知分组结构: 整体视为一组, replicate_index 按顺序编号。
        plates["plate_group"] = 1
        plates["replicate_index"] = range(1, n_plates + 1)
    return plates.drop(columns=["_num"])


def read_ic50_matrix(exp_dir: Path, plate_id: str) -> pd.DataFrame | None:
    path = exp_dir / "data" / f"ABconc_with_exptime_plate{plate_id}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.columns = [str(c).strip() for c in df.columns]
    if TIME_COL not in df.columns:
        raise SystemExit(f"ERROR: {path} 缺少时间列 {TIME_COL}")
    return df


def read_growth_matrix(exp_dir: Path, plate_id: str) -> pd.DataFrame | None:
    matches = sorted((exp_dir / "data").glob(f"growth_rates_*_plate{plate_id}.csv"))
    if not matches:
        return None
    df = pd.read_csv(matches[0])
    first = df.columns[0]
    df = df.set_index(first)
    df.columns = [str(c).strip().replace("_gr_fit", "") for c in df.columns]
    return df


def resolve_background(code: str, gpm_map: dict[str, str] | None) -> tuple[str, str, bool]:
    """把布局编码解析为 (background, well_role, resolved)。"""
    text = str(code).strip()
    if text.upper() in BLANK_CODES:
        return "", "blank", True
    if text.lower().startswith("laca"):
        # ΔlacA 是论文的参考株 (携带同样的卡那抗性 cassette, 但 lacA 与耐药无关)。
        return "lacA", "reference", True
    if gpm_map is not None and text in gpm_map:
        return gpm_map[text], "deletion", True
    # UTI 分离株 (R1..R7) 或未知编码: 保留原始编码, 标记为未解析。
    role = "isolate" if re.match(r"^R\d+$", text) else "unknown"
    return text, role, False


def build_experiment(
    fsepi_dir: Path,
    cfg: dict,
    gpm_maps: dict[int, dict[str, str]],
) -> tuple[pd.DataFrame, dict]:
    exp_dir = fsepi_dir / cfg["dir"]
    if not exp_dir.is_dir():
        return pd.DataFrame(), {"experiment": cfg["dir"], "status": "missing"}

    plates = assign_plate_groups(discover_plates(exp_dir), cfg["n_plate_groups"])
    records: list[pd.DataFrame] = []
    n_missing_ic50 = 0
    n_unresolved = 0

    for _, plate in plates.iterrows():
        plate_id = plate["plate_id"]
        ic50 = read_ic50_matrix(exp_dir, plate_id)
        if ic50 is None:
            n_missing_ic50 += 1
            continue
        growth = read_growth_matrix(exp_dir, plate_id)
        layout = parse_layout(plate["layout_path"])
        ab_layout = (
            parse_layout(plate["antibiotic_layout_path"])
            if isinstance(plate["antibiotic_layout_path"], Path)
            else None
        )
        # 9 板实验: 同一 plate_group 共享一份 GPm 映射。
        gpm_idx = int(plate["plate_group"]) if cfg["n_plate_groups"] else None
        gpm_map = gpm_maps.get(gpm_idx) if gpm_idx else None

        time_h = pd.to_numeric(ic50[TIME_COL], errors="coerce")
        for well, code in layout.items():
            if well not in ic50.columns:
                continue
            background, role, resolved = resolve_background(code, gpm_map)
            if role == "blank":
                continue
            if not resolved:
                n_unresolved += 1
            antibiotic = cfg["antibiotic"]
            if ab_layout is not None:
                antibiotic = ab_layout.get(well) or antibiotic
            if not antibiotic:
                continue

            values = pd.to_numeric(ic50[well], errors="coerce")
            frame = pd.DataFrame({
                "experiment": cfg["dir"],
                "antibiotic": antibiotic,
                "plate_id": plate_id,
                "plate_index": int(plate["plate_index"]),
                "plate_group": int(plate["plate_group"]),
                "well": well,
                "strain_code": code,
                "background": background,
                "well_role": role,
                "background_resolved": bool(resolved),
                "replicate_index": int(plate["replicate_index"]),
                "dilution_number": ic50.index.astype(str),
                "time_h": time_h.to_numpy(),
                "ic50": values.to_numpy(),
            })
            if growth is not None and well in growth.columns:
                # 生长速率文件通常比 IC50 少一个稀释循环, 按位置对齐前 n 行。
                gr = pd.to_numeric(growth[well], errors="coerce").to_numpy(dtype=float)
                aligned = np.full(len(ic50), np.nan)
                n = min(gr.size, len(ic50))
                aligned[:n] = gr[:n]
                frame["growth_rate"] = aligned
            else:
                frame["growth_rate"] = np.nan
            records.append(frame)

    if not records:
        return pd.DataFrame(), {"experiment": cfg["dir"], "status": "no_data"}

    out = pd.concat(records, ignore_index=True)
    out["strain_group_id"] = out["antibiotic"] + ":" + out["well"] + "_" + out["background"]
    out["population_id"] = (
        out["antibiotic"] + ":" + out["plate_id"] + ":" + out["well"] + "_" + out["background"]
    )
    manifest = {
        "experiment": cfg["dir"],
        "status": "ok",
        "n_plates": int(len(plates)),
        "n_plates_missing_ic50": int(n_missing_ic50),
        "n_populations": int(out["population_id"].nunique()),
        "n_strain_groups": int(out["strain_group_id"].nunique()),
        "n_backgrounds": int(out.loc[out["well_role"] != "blank", "background"].nunique()),
        "n_timepoints_median": float(out.groupby("population_id").size().median()),
        "antibiotics": sorted(out["antibiotic"].dropna().unique().tolist()),
        "n_wells_unresolved_background": int(n_unresolved),
        "plate_group_to_gpm": (
            {str(g): f"GPm{g}" for g in sorted(plates["plate_group"].unique())}
            if cfg["n_plate_groups"] else None
        ),
    }
    return out, manifest


def summarise_populations(traj: pd.DataFrame) -> pd.DataFrame:
    """每个 population 一行: 初始/终点 IC50、fold increase、AUC、初始生长速率。"""
    rows = []
    for pid, sub in traj.groupby("population_id", sort=False):
        sub = sub.sort_values("time_h")
        ic = sub["ic50"].to_numpy(dtype=float)
        t = sub["time_h"].to_numpy(dtype=float)
        valid = np.isfinite(ic) & np.isfinite(t)
        ic_v, t_v = ic[valid], t[valid]
        # 初始 IC50 取前若干个非零读数的中位数: 循环 0 的浓度恒为 0 (尚未加药)。
        nonzero = ic_v[ic_v > 0]
        initial = float(np.median(nonzero[:3])) if nonzero.size else math.nan
        final = float(np.median(ic_v[-3:])) if ic_v.size else math.nan
        head = sub.iloc[0]
        rows.append({
            "population_id": pid,
            "strain_group_id": head["strain_group_id"],
            "experiment": head["experiment"],
            "antibiotic": head["antibiotic"],
            "plate_id": head["plate_id"],
            "plate_group": head["plate_group"],
            "well": head["well"],
            "background": head["background"],
            "well_role": head["well_role"],
            "replicate_index": head["replicate_index"],
            "n_timepoints": int(valid.sum()),
            "duration_h": float(t_v.max() - t_v.min()) if t_v.size else math.nan,
            "initial_ic50": initial,
            "final_ic50": final,
            "max_ic50": float(np.nanmax(ic_v)) if ic_v.size else math.nan,
            "fold_increase": (final / initial) if initial and initial > 0 else math.nan,
            # AUC 在 log10(IC50) 上按时间积分, 是论文比较轨迹整体高低的量 (Fig 4a 的打分基础)。
            "log10_ic50_auc": (
                float(np.trapezoid(np.log10(np.clip(ic_v, 1e-6, None)), t_v))
                if ic_v.size > 1 else math.nan
            ),
            "initial_growth_rate": (
                float(np.nanmedian(sub["growth_rate"].to_numpy(dtype=float)[:3]))
                if sub["growth_rate"].notna().any() else math.nan
            ),
        })
    return pd.DataFrame(rows)


def attach_paper_reference(
    pops: pd.DataFrame,
    dataset_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    """接入论文自带的权威初始/终点耐药值, 并与本管线的计算结果互校。

    为什么需要: morbidostat 的反馈回路是从 0 浓度逐步爬坡的, 因此 ABconc 时间序列
    最开始的几个非零读数是爬坡伪影, 不能当作祖先的 IC50。论文的初始耐药来自独立的
    dose-response 测量, 存放在 `FigS5_and_FigS2` 的 `iniIC50` 列。

    - FigS5_and_FigS2 : 每 (抗生素, 菌株) 的 iniIC50 -> initial_ic50_paper
    - Fig2a_violin    : 每 (抗生素, plate, well) 的终点 IC50 与 fold_increase -> 校验用
    """
    report: dict = {"status": "ok"}
    main_x = dataset_dir / "SourceData_MainFigures.xlsx"
    supp_x = dataset_dir / "SourceData_SupplementaryFigures.xlsx"
    if not (main_x.exists() and supp_x.exists()):
        return pops, {"status": "source_data_missing"}

    ini = pd.read_excel(supp_x, sheet_name="FigS5_and_FigS2")
    ini = ini.rename(columns={"ab": "antibiotic", "strain": "background", "iniIC50": "initial_ic50_paper"})
    ini = (
        ini.groupby(["antibiotic", "background"], as_index=False)["initial_ic50_paper"].median()
    )

    end = pd.read_excel(main_x, sheet_name="Fig2a_violin")
    end = end.rename(columns={
        "ab": "antibiotic", "plate": "plate_id", "IC50": "final_ic50_paper",
        "fold_increase": "fold_increase_paper", "time[h]_final": "final_time_h_paper",
    })
    end["plate_id"] = end["plate_id"].astype(str)
    end = end[["antibiotic", "plate_id", "well", "final_ic50_paper",
               "fold_increase_paper", "final_time_h_paper"]].drop_duplicates(
        subset=["antibiotic", "plate_id", "well"])

    out = pops.copy()
    out["plate_id"] = out["plate_id"].astype(str)
    out = out.merge(ini, on=["antibiotic", "background"], how="left")
    out = out.merge(end, on=["antibiotic", "plate_id", "well"], how="left")

    # 用论文的初始耐药重算 fold increase; 保留本管线的爬坡式估计以便对比。
    out = out.rename(columns={"initial_ic50": "initial_ic50_rampup",
                              "fold_increase": "fold_increase_rampup"})
    out["initial_ic50"] = out["initial_ic50_paper"]
    out["fold_increase"] = out["final_ic50"] / out["initial_ic50"]

    both = out.dropna(subset=["final_ic50", "final_ic50_paper"])
    if len(both) > 1:
        report["n_populations_with_paper_endpoint"] = int(len(both))
        report["final_ic50_log10_corr"] = round(float(np.corrcoef(
            np.log10(np.clip(both["final_ic50"], 1e-9, None)),
            np.log10(np.clip(both["final_ic50_paper"], 1e-9, None)),
        )[0, 1]), 4)
        report["final_ic50_median_ratio"] = round(
            float((both["final_ic50"] / both["final_ic50_paper"]).median()), 4)
    ref = out[(out["well_role"] == "reference") & out["experiment"].str.contains("9plates")]
    report["reference_strain_median_fold_increase"] = {
        str(ab): round(float(v), 1)
        for ab, v in ref.groupby("antibiotic")["fold_increase"].median().items()
    }
    report["paper_reference_strain_fold_increase"] = {"MEC": 390, "NIT": 70, "TMP": 302}
    report["n_missing_initial_ic50"] = int(out["initial_ic50"].isna().sum())
    return out, report


def crosscheck_genotype(pops: pd.DataFrame, genotype_manifest: Path) -> dict:
    """用基因型 manifest 校验 plate/well -> background 映射是否一致。

    GPm 文件使用旧基因名 (如 yfgA/rfaC/erfK/gntY), 而 SourceData 使用现名
    (rodZ/waaC/ldtA/nfuA)。这类差异的特征是: 同一孔在其全部 replicate 板上给出
    完全一致的配对, 且两侧名字构成一一对应。据此把"同义名"与"真正的映射错误"区分开。
    """
    if not genotype_manifest.exists():
        return {"status": "genotype_manifest_missing", "path": str(genotype_manifest)}
    geno = pd.read_csv(genotype_manifest, dtype=str)
    needed = {"plate_id", "well", "background"}
    if not needed.issubset(geno.columns):
        return {"status": "genotype_manifest_incompatible", "missing": sorted(needed - set(geno.columns))}
    key = ["plate_id", "well"]
    left = geno[key + ["background"]].drop_duplicates()
    right = pops[key + ["background"]].drop_duplicates().rename(columns={"background": "background_phenotype"})
    merged = left.merge(right, on=key, how="left")
    resolvable = int(merged["background_phenotype"].notna().sum())
    exact = int((merged["background"] == merged["background_phenotype"]).sum())

    diff = merged[
        merged["background_phenotype"].notna()
        & (merged["background"] != merged["background_phenotype"])
    ]
    pairs = diff[["background", "background_phenotype"]].drop_duplicates()
    # 一一对应 => 同义名; 一对多/多对一 => 真正的映射错误。
    bijective = (
        pairs["background"].is_unique and pairs["background_phenotype"].is_unique
    )
    synonym_pairs = (
        [{"source_data": a, "gpm_layout": b} for a, b in pairs.itertuples(index=False)]
        if bijective else []
    )
    n_synonym = int(len(diff)) if bijective else 0
    return {
        "status": "ok",
        "n_genotype_wells": int(len(merged)),
        "n_found_in_phenotype": resolvable,
        "n_exact_match": exact,
        "n_synonym_match": n_synonym,
        "n_true_mismatch": resolvable - exact - n_synonym,
        "match_rate_including_synonyms": (
            round((exact + n_synonym) / resolvable, 4) if resolvable else None
        ),
        "synonym_pairs": synonym_pairs,
        "examples_true_mismatch": (
            [] if bijective else diff.head(10).to_dict(orient="records")
        ),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build phenotype (IC50) evolution trajectories for ecoli2026")
    ap.add_argument("--dataset-dir", type=Path, default=default_dataset_dir(), help="`ecoli2026` 数据目录")
    ap.add_argument("--outdir", type=Path, default=default_outdir(), help="输出目录")
    ap.add_argument("--genotype-manifest", type=Path, default=None,
                    help="用于交叉校验的基因型 sample_manifest.csv (默认 derived/genotype_route_b/sample_manifest.csv)")
    args = ap.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    fsepi_dir = dataset_dir / "fsepi_data"
    if not fsepi_dir.is_dir():
        raise SystemExit(f"ERROR: 找不到 fsepi_data: {fsepi_dir}")
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    genotype_manifest = args.genotype_manifest or (
        dataset_dir / "derived" / "genotype_route_b" / "sample_manifest.csv"
    )

    gpm_maps = load_gpm_maps(fsepi_dir)
    parts: list[pd.DataFrame] = []
    manifests: list[dict] = []
    for cfg in EXPERIMENTS:
        traj, manifest = build_experiment(fsepi_dir, cfg, gpm_maps)
        manifests.append(manifest)
        if not traj.empty:
            parts.append(traj)
        print("[ecoli][phenotype] %s: %s" % (cfg["dir"], json.dumps(manifest, ensure_ascii=False)))

    if not parts:
        raise SystemExit("ERROR: 没有解析到任何表型轨迹")

    traj = pd.concat(parts, ignore_index=True)
    traj = traj.sort_values(["antibiotic", "strain_group_id", "replicate_index", "time_h"]).reset_index(drop=True)
    pops = summarise_populations(traj)
    pops, paper_report = attach_paper_reference(pops, dataset_dir)

    traj_path = outdir / "phenotype_trajectories.csv"
    pops_path = outdir / "phenotype_populations.csv"
    traj.to_csv(traj_path, index=False)
    pops.to_csv(pops_path, index=False)

    payload = {
        "source": "fsepi_data morbidostat runs (ABconc_with_exptime_* = per-well IC50 time series)",
        "has_temporal_axis": True,
        "unit_of_observation": "population x dilution cycle",
        "assumptions": [
            "ic50 = antibiotic concentration maintained by the feedback loop at 50% growth inhibition",
            "plate groups of equal size along ascending plate_id are replicate sets sharing one GPm layout",
            "final_ic50 = median of last 3 readouts of the ABconc time series",
            "initial_ic50 = paper's independently measured iniIC50 (FigS5_and_FigS2); the ramp-up based "
            "estimate is kept separately as initial_ic50_rampup and must NOT be used as ancestor IC50",
            "log10_ic50_auc integrates log10(IC50) over time (basis of the paper's trajectory deviation score)",
        ],
        "experiments": manifests,
        "paper_reference_crosscheck": paper_report,
        "genotype_crosscheck": crosscheck_genotype(pops, genotype_manifest),
        "global": {
            "n_rows": int(len(traj)),
            "n_populations": int(traj["population_id"].nunique()),
            "n_strain_groups": int(traj["strain_group_id"].nunique()),
            "n_backgrounds": int(traj["background"].nunique()),
            "antibiotics": sorted(traj["antibiotic"].dropna().unique().tolist()),
        },
    }
    with open(outdir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print(
        "[ecoli][phenotype] rows=%d populations=%d strain_groups=%d backgrounds=%d -> %s"
        % (
            len(traj), traj["population_id"].nunique(), traj["strain_group_id"].nunique(),
            traj["background"].nunique(), traj_path,
        )
    )
    print("[ecoli][phenotype] paper crosscheck: %s"
          % json.dumps(paper_report, ensure_ascii=False))
    print("[ecoli][phenotype] genotype crosscheck: %s"
          % json.dumps(
              {k: payload["genotype_crosscheck"].get(k) for k in
               ("n_exact_match", "n_synonym_match", "n_true_mismatch",
                "match_rate_including_synonyms", "synonym_pairs")},
              ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
