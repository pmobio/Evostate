#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
谱系分组 CV 评估：B0/B1 基线、可选 HyperTraPS、可选自研 DL。
- 对每个 fold: 整个 group 进出，防止近缘泄漏
- 主指标: clean 单事件边上的 mean log-likelihood / Top-k / MRR / AUC
- 辅助: multi-event 边上的 multi-Top-k
- 统计: paired bootstrap on per-item log-likelihood
- 过拟合诊断: 若传入 --dl-*, 每折在 train 上重新 fit 一个 DL, 并报告 fit_report

双通道 (方案B) gain + loss 评估:
- gain 侧候选 = s_i==0, 真值 = gained; loss 侧候选 = s_i==1 且属可逆先验子集,
  真值 = lost。两侧各自独立归一化并分别报告。
- gain 侧口径与历史 gain-only 结果完全一致 (JSON 中 metrics/overall 仍为 gain 侧),
  loss 侧作为新增能力写入 loss_side_eval, 只与其自身基线 (uniform / loss-prevalence)
  对照, 用于回答"引入 loss 后新增的可逆路径能否被正常预测"; 两侧指标不作并列比较。
- 两侧共用同一套 fold 划分 (按两侧评估边的并集构造), 以便 gain/loss 指标可联合解读。
- HyperTraPS 原理上只支持单调累积, 因此不参与 loss 侧评估; 其训练输入也始终只用
  含 gain 的边。
- loss 侧的 1->0 解释为"端点净状态消失/回退", 不等于已确证的回复突变事件。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(ROOT, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

import model_dl as MDL
import train_dl as TDL
from baselines import (
    HyperTraPSThetaBaseline,
    LossPrevalenceBaseline,
    PrevalenceBaseline,
    UniformBaseline,
)
from evaluator import (
    estimate_gain_event_share,
    eval_transitions,
    eval_transitions_combined,
    group_kfold_indices,
    paired_bootstrap_logl,
    random_kfold_indices,
    summarize,
)

MAIN_METRICS = ["mean_logl", "top1", "top3", "top5", "mrr", "auc"]
EXTRA_METRICS = [
    "multi_top1", "multi_top3", "multi_top5", "n_clean", "n_multi",
    # logl_lift_vs_uniform = mean_logl - 同侧同折 B0_uniform 的 mean_logl,
    # 即"相对该侧随机下界的信息增益 (nats)"。
    # 用途: 在同一侧内做跨模型/跨折比较时消除候选集规模波动带来的平移。
    # 注意: gain 与 loss 两侧候选空间规模差可达两个数量级 (gain 候选 ~L,
    # loss 候选仅当前已存在且可逆的位), 两侧数值不作并列比较 —— loss 侧只与
    # 其自身基线 (uniform / loss-prevalence) 对照, 用于判断可逆路径是否被学到。
    "logl_lift_vs_uniform",
    "mean_n_candidates",
    # 全边/事件级指标: 不依赖"单事件干净边"。多事件数据上 clean 口径样本量可能为
    # 个位数甚至 0 (combined 通道尤其明显), 此时应以这组指标为主。
    "auc_all", "event_mrr",
    "prec1", "prec3", "prec5",
    "rec1", "rec3", "rec5",
    "n_events_scored", "n_edges_scored",
]

GAIN = "gain"
LOSS = "loss"
COMBINED = "combined"
# loss 侧不参与的方法 (原理上只支持单调累积)
LOSS_UNSUPPORTED_PREFIXES = ("HT_",)


def describe_compute_environment() -> Dict[str, Any]:
    """记录计算环境, 用于方法间时间对比的公平性声明。

    时间对比只有在设备口径明确时才有意义: HyperTraPS 为 CPU MCMC (由
    --ht-threads 控制线程数), 而 DL 模型走 PyTorch。`model_dl.SetPredDL`
    当前固定在 CPU 上训练 (见 models/model_dl.py 中 dev=torch.device("cpu")),
    因此本脚本内的 DL vs HyperTraPS 比较默认是同设备 (CPU) 比较。
    """
    info: Dict[str, Any] = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "torch_available": False,
        "torch_version": None,
        "torch_num_threads": None,
        "cuda_available": False,
        "gpu_names": [],
        "dl_device": None,
        "note": None,
    }
    try:
        import torch  # noqa: WPS433 (运行期可选依赖)

        info["torch_available"] = True
        info["torch_version"] = str(torch.__version__)
        info["torch_num_threads"] = int(torch.get_num_threads())
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu_names"] = [
                str(torch.cuda.get_device_name(i)) for i in range(torch.cuda.device_count())
            ]
    except Exception as exc:  # pragma: no cover - torch 为可选依赖
        info["note"] = f"torch 不可用: {exc}"

    # SetPredDL 固定使用 CPU, 这里显式记录以免误读加速比
    dl_device = "cpu"
    try:
        dl_device = str(getattr(MDL, "DL_DEVICE", "cpu"))
    except Exception:
        dl_device = "cpu"
    info["dl_device"] = dl_device
    return info


def _timed(fn, *args, **kwargs) -> Tuple[Any, float]:
    """执行 fn 并返回 (结果, 墙钟秒数)。"""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def summarize_runtime(fold_runtimes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """把逐折计时汇总为每方法的 fit / predict / total 秒数。"""
    agg: Dict[str, Dict[str, List[float]]] = {}
    for row in fold_runtimes:
        for method, phases in row.get("methods", {}).items():
            slot = agg.setdefault(method, {"fit_sec": [], "predict_sec": []})
            for phase in ("fit_sec", "predict_sec"):
                val = phases.get(phase)
                if val is not None:
                    slot[phase].append(float(val))

    out: Dict[str, Any] = {}
    for method, phases in agg.items():
        fit = phases["fit_sec"]
        pred = phases["predict_sec"]
        total = [a + b for a, b in zip(fit, pred)] if len(fit) == len(pred) else (fit or pred)
        entry = {
            "n_folds_timed": int(max(len(fit), len(pred))),
            "fit_sec_total": (float(np.sum(fit)) if fit else None),
            "fit_sec_mean": (float(np.mean(fit)) if fit else None),
            "predict_sec_total": (float(np.sum(pred)) if pred else None),
            "predict_sec_mean": (float(np.mean(pred)) if pred else None),
            "total_sec": (float(np.sum(total)) if total else None),
        }
        out[method] = entry
    return out
ALL_METRICS = MAIN_METRICS + EXTRA_METRICS


def _nanmean(vals):
    arr = np.asarray(vals, dtype=float)
    if arr.size == 0:
        return float("nan")
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _jsonable(obj: Any):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return None if math.isnan(x) else x
    if isinstance(obj, Path):
        return str(obj)
    return obj


def transition_profile(trans_all: List[dict], trans_eval: List[dict],
                       sides: List[str] | None = None) -> Dict[str, Any]:
    single_gain = sum(1 for t in trans_all if len(t["gained"]) == 1 and len(t["lost"]) == 0)
    multi_gain = sum(1 for t in trans_all if len(t["gained"]) > 1 and len(t["lost"]) == 0)
    mixed_gain_loss = sum(1 for t in trans_all if len(t["gained"]) > 0 and len(t["lost"]) > 0)
    loss_only = sum(1 for t in trans_all if len(t["gained"]) == 0 and len(t["lost"]) > 0)
    no_change = sum(1 for t in trans_all if len(t["gained"]) == 0 and len(t["lost"]) == 0)
    strategies = sorted({str(t.get("transition_strategy", "")).strip() for t in trans_all if str(t.get("transition_strategy", "")).strip()})
    n_gain_edges = sum(1 for t in trans_all if len(t["gained"]) > 0)
    n_loss_edges = sum(1 for t in trans_all if len(t["lost"]) > 0)
    return {
        "strategies": strategies,
        "eval_sides": list(sides or [GAIN]),
        "n_input_transitions": int(len(trans_all)),
        "n_eval_transitions": int(len(trans_eval)),
        # 历史字段: gain-only 口径下被跳过的边数
        "n_skipped_non_gain": int(len(trans_all) - n_gain_edges),
        "n_gain_side_edges": int(n_gain_edges),
        "n_loss_side_edges": int(n_loss_edges),
        "single_gain": int(single_gain),
        "multi_gain": int(multi_gain),
        "mixed_gain_loss": int(mixed_gain_loss),
        "loss_only": int(loss_only),
        "no_change": int(no_change),
        "single_loss": int(sum(1 for t in trans_all if len(t["lost"]) == 1)),
        "multi_loss": int(sum(1 for t in trans_all if len(t["lost"]) > 1)),
    }


def aggregate_fold_metrics(folds_payload: List[dict], method_labels: List[str],
                           methods_key: str = "methods") -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    metrics_nested: Dict[str, Dict[str, float]] = {m: {} for m in MAIN_METRICS}
    overall_by_method: Dict[str, Dict[str, float]] = {}
    for method in method_labels:
        rows = [fold[methods_key][method] for fold in folds_payload
                if method in fold.get(methods_key, {})]
        if not rows:
            continue
        overall_by_method[method] = {}
        for metric in ALL_METRICS:
            val = _nanmean([row.get(metric, float("nan")) for row in rows])
            overall_by_method[method][metric] = val
            if metric in MAIN_METRICS:
                metrics_nested[metric][method] = val
    return metrics_nested, overall_by_method


def verdict_text(delta: float, p_value: float, left_name: str, right_name: str) -> str:
    if p_value is None or p_value >= 0.05:
        return "not significant"
    return f"{left_name} better" if delta > 0 else f"{right_name} better"


def compare_methods(per_item: Dict[str, List[np.ndarray]], left_key: str, right_key: str,
                    left_name: str, right_name: str, seed: int) -> Dict[str, Any] | None:
    if left_key not in per_item or right_key not in per_item:
        return None
    left = np.concatenate(per_item[left_key]) if per_item[left_key] else np.array([], dtype=float)
    right = np.concatenate(per_item[right_key]) if per_item[right_key] else np.array([], dtype=float)
    if len(left) == 0 or len(right) == 0:
        return None
    n = min(len(left), len(right))
    delta, p_value = paired_bootstrap_logl(left[:n], right[:n], n_boot=2000, seed=seed)
    return {
        "left_method": left_key,
        "right_method": right_key,
        "delta_mean_logl": float(delta),
        "p_value": float(p_value),
        "verdict": verdict_text(float(delta), float(p_value), left_name, right_name),
    }


def stratum_of(t: dict, mode: str) -> str | None:
    """取一条转移的分层标签。

    "group_prefix": 取 group_id 中第一个 ':' 之前的部分。ecoli2026 Route-B 的
    group_id 形如 'MEC:A10_pntB', 前缀即施加的抗生素 (选择压力)。
    """
    if mode == "none":
        return None
    if mode == "group_prefix":
        g = str(t.get("group", ""))
        return g.split(":")[0] if ":" in g else g
    raise ValueError(f"未知 --stratify-by 模式: {mode}")


def resolve_stratify_mode(trans_eval: List[dict], mode: str) -> Tuple[str, Dict[str, Any]]:
    """决定实际使用的分层模式。

    auto: 仅当所有 group_id 都含 ':' 且不同前缀数在 [2,10] 时启用 group_prefix。
    这样 ecoli2026 (MEC/NIT/TMP) 会自动分层, 而 TB / Kleborate22 / AMRGlobal16
    的 group_id (谱系 / ST / cgMLST 簇) 不含 ':' -> 自动保持不分层, 行为不变。
    """
    info: Dict[str, Any] = {"requested": mode, "used": "none", "labels": [], "reason": None}
    if mode == "none":
        info["reason"] = "disabled"
        return "none", info

    groups = [str(t.get("group", "")) for t in trans_eval]
    all_have_sep = bool(groups) and all(":" in g for g in groups)
    prefixes = sorted({g.split(":")[0] for g in groups if ":" in g})

    if mode == "group_prefix":
        if not all_have_sep:
            raise SystemExit(
                "--stratify-by group_prefix 需要所有 group_id 均包含 ':' 分隔符")
        info.update({"used": "group_prefix", "labels": prefixes, "reason": "explicit"})
        return "group_prefix", info

    # auto
    if all_have_sep and 2 <= len(prefixes) <= 10:
        info.update({
            "used": "group_prefix",
            "labels": prefixes,
            "reason": f"auto: 所有 group_id 含 ':' 且前缀数={len(prefixes)}",
        })
        return "group_prefix", info
    info["reason"] = (
        "auto: group_id 不含 ':' 或前缀数超出 [2,10], 不分层"
        if not all_have_sep else f"auto: 前缀数={len(prefixes)} 超出 [2,10], 不分层"
    )
    return "none", info


def aggregate_strata(folds_payload: List[dict], labels: List[str],
                     method_orders: Dict[str, List[str]]) -> Dict[str, Any]:
    """把逐折的分层结果聚合成 {stratum: {channel: {"overall":..., "metrics":...}}}。"""
    out: Dict[str, Any] = {}
    for label in labels:
        entry: Dict[str, Any] = {
            "n_test_per_fold": [
                (f.get("strata", {}).get(label) or {}).get("n_test") for f in folds_payload
            ],
            "pi_gain_per_fold": [
                (f.get("strata", {}).get(label) or {}).get("pi_gain") for f in folds_payload
            ],
        }
        for channel, key in (("gain", "methods"), ("loss", "methods_loss"),
                             ("combined", "methods_combined")):
            sub_folds = [
                {key: (f.get("strata", {}).get(label) or {}).get(key, {})}
                for f in folds_payload
            ]
            if not any(sf[key] for sf in sub_folds):
                continue
            metrics, overall = aggregate_fold_metrics(
                sub_folds, method_orders[channel], methods_key=key)
            entry[channel] = {"metrics": metrics, "overall": overall}
        out[label] = entry
    return out


def attach_logl_lift(methods_payload: Dict[str, Dict[str, float]],
                     baseline: str = "B0_uniform") -> None:
    """就地写入 logl_lift_vs_uniform = mean_logl - 同侧同折 uniform 的 mean_logl。

    mean_logl 的绝对值随该侧候选空间大小平移, 减掉同侧同折的随机下界后, 同一侧内
    的跨模型/跨折比较更稳定。两侧数值不作并列比较。
    """
    base = methods_payload.get(baseline, {}).get("mean_logl")
    if base is None or (isinstance(base, float) and math.isnan(base)):
        return
    for payload in methods_payload.values():
        val = payload.get("mean_logl")
        if val is None or (isinstance(val, float) and math.isnan(val)):
            payload["logl_lift_vs_uniform"] = float("nan")
        else:
            payload["logl_lift_vs_uniform"] = float(val) - float(base)


def eval_fold_strata(test: List[dict], train: List[dict], stratify_mode: str,
                     predictors: Dict[str, Any], predictors_loss: Dict[str, Any],
                     method_order: List[str], loss_method_order: List[str],
                     combined_method_order: List[str], sides: List[str],
                     loss_mask, uniform_fn, pooled_pi: float | None = None,
                     topk_eps: float = 1e-12) -> Dict[str, Any]:
    """在一折内按分层标签分别评估 gain / loss / combined 三个通道。

    fold 划分本身保持池化 (不按分层切分), 只在评估阶段拆分测试集, 因此:
      - 训练数据不变, 模型仍是单一池化模型 (药物等环境变量不进模型);
      - 各层指标可直接与池化指标对照, 用于暴露层间差异。

    combined 通道的 side 权重 pi 优先用"该层训练子集"估计 —— 层间 gain/loss 平衡
    可能显著不同 (如 ecoli MEC 的 pi<0.5 而 NIT>0.5), 用池化 pi 会使该层有偏。
    该层训练事件过少 (<20) 时回退到池化 pi, 并在 pi_source 中标注。
    """
    if stratify_mode == "none":
        return {}

    labels = sorted({stratum_of(t, stratify_mode) for t in test} - {None})
    out: Dict[str, Any] = {}
    for label in labels:
        test_s = [t for t in test if stratum_of(t, stratify_mode) == label]
        train_s = [t for t in train if stratum_of(t, stratify_mode) == label]
        if not test_s:
            continue
        entry: Dict[str, Any] = {
            "n_test": int(len(test_s)),
            "n_train": int(len(train_s)),
            "n_test_gain": int(sum(1 for t in test_s if len(t.get("gained", []) or []) > 0)),
            "n_test_loss": int(sum(1 for t in test_s if len(t.get("lost", []) or []) > 0)),
            "methods": {},
        }
        for method in method_order:
            if method not in predictors:
                continue
            ev = eval_transitions(test_s, predictors[method], side=GAIN, eps=topk_eps)
            entry["methods"][method] = _jsonable(summarize(ev))
        attach_logl_lift(entry["methods"])

        if LOSS in sides:
            entry["methods_loss"] = {}
            for method in loss_method_order:
                if method not in predictors_loss:
                    continue
                ev_l = eval_transitions(test_s, predictors_loss[method], side=LOSS,
                                        loss_mask=loss_mask, eps=topk_eps)
                entry["methods_loss"][method] = _jsonable(summarize(ev_l))
            attach_logl_lift(entry["methods_loss"])

            n_events_s = sum(
                len(t.get("gained", []) or []) + len(t.get("lost", []) or [])
                for t in train_s
            )
            if n_events_s >= 20:
                pi_s = estimate_gain_event_share(train_s)
                entry["pi_source"] = "stratum_train"
            else:
                pi_s = (pooled_pi if pooled_pi is not None
                        else estimate_gain_event_share(train))
                entry["pi_source"] = "pooled_fallback(n_events<20)"
            entry["pi_gain"] = float(pi_s)
            entry["n_train_events"] = int(n_events_s)

            entry["methods_combined"] = {}
            for method in combined_method_order:
                if method not in predictors:
                    continue
                lfn = predictors_loss.get(method, uniform_fn)
                ev_c = eval_transitions_combined(
                    test_s, predictors[method], loss_predict_fn=lfn,
                    loss_mask=loss_mask, pi=pi_s, eps=topk_eps)
                entry["methods_combined"][method] = _jsonable(summarize(ev_c))
            attach_logl_lift(entry["methods_combined"])

        out[label] = entry
    return out


def fmt_metrics(name: str, s: Dict[str, float], side: str | None = None) -> str:
    tag = name if side is None else f"[{side}]{name}"

    def _f(key: str) -> float:
        """取数值; 缺失或 None 一律当 NaN。

        指标 payload 可能已过 _jsonable (NaN -> None), 例如某折在某一侧没有
        clean 事件边时。这里统一兜底, 避免格式化 None 抛 TypeError。
        """
        val = s.get(key)
        if val is None:
            return float("nan")
        try:
            return float(val)
        except (TypeError, ValueError):
            return float("nan")

    return (
        f"{tag:>22s} | "
        f"logL={_f('mean_logl'):.4f} | "
        f"top1={_f('top1'):.3f} "
        f"top3={_f('top3'):.3f} "
        f"top5={_f('top5'):.3f} | "
        f"MRR={_f('mrr'):.3f} | "
        f"AUC={_f('auc'):.3f} | "
        f"lift={_f('logl_lift_vs_uniform'):+.3f} | "
        f"cand={_f('mean_n_candidates'):.1f} | "
        f"clean={int(round(s.get('n_clean', 0) or 0))} "
        f"multi={int(round(s.get('n_multi', 0) or 0))} "
        # 全边口径 (主指标, 样本量 = n_edges/n_events, 远大于 clean)
        f"|| aucA={_f('auc_all'):.3f} "
        f"eMRR={_f('event_mrr'):.3f} "
        f"rec5={_f('rec5'):.3f} "
        f"nE={int(round(s.get('n_events_scored', 0) or 0))}"
    )


def load_hierarchy_context(args, transitions_path: str, trans_all: List[dict],
                           L: int) -> Dict[str, Any]:
    """定位并加载 feature_hierarchy.csv 一次, 供 metadata / loss mask 共用。

    位序对齐校验在这里统一做一次, 保证 evaluator 的 loss 候选掩码与 DL 的
    loss head 掩码来自同一份、同一位序的先验。
    """
    hpath = args.hierarchy or TDL.find_hierarchy_path(transitions_path)
    hdf = None
    feat_ids = None
    n_ok = 0
    if hpath and os.path.isfile(hpath):
        hdf = TDL.load_feature_hierarchy(hpath, L)
        feat_ids = hdf["feature_id"].tolist()
        n_ok = TDL.validate_feature_alignment(trans_all, feat_ids)
        print(f"[phylo_eval] hierarchy 对齐校验通过: {n_ok} 条 ({hpath})")
    return {
        "path": (None if hpath is None else os.path.abspath(hpath)),
        "exists": hdf is not None,
        "hdf": hdf,
        "feat_ids": feat_ids,
        "n_alignment_checked": int(n_ok),
    }


def resolve_loss_eval_mask(hctx: Dict[str, Any], trans_info: dict) -> Tuple[Any, Dict[str, Any]]:
    """从 hierarchy 的可逆先验构造 loss 侧候选掩码 (evaluator 与 DL 共用)。

    返回 (mask 或 None, info)。mask 为 None 表示 loss 侧不可用。
    """
    info: Dict[str, Any] = {
        "hierarchy": hctx["path"],
        "n_alignment_checked": hctx["n_alignment_checked"],
        "n_loss_total": int(trans_info["n_loss_total"]),
        "n_loss_transitions": int(trans_info["n_loss_transitions"]),
        "used": False,
        "reason": None,
        "source": None,
        "n_loss_eligible_features": 0,
        "loss_eligible_features": [],
    }
    if hctx["hdf"] is None:
        info["reason"] = "未找到 feature_hierarchy.csv, 无法界定 loss 侧候选集合"
        return None, info

    mask, scan = TDL.load_selective_loss_mask(hctx["hdf"])
    info.update(scan)
    if int(mask.sum()) <= 0:
        info["used"] = False
        info["reason"] = scan.get("reason") or "hierarchy 中未找到可逆先验"
        return None, info
    info["used"] = True
    info["reason"] = "ok"
    return mask, info


def prepare_dl_setup(args, hctx: Dict[str, Any], trans_info: dict,
                     requested_meta_fields: List[str] | None,
                     loss_mask=None, loss_scan: Dict[str, Any] | None = None) -> Dict[str, Any]:
    hpath = hctx["path"]
    hdf = hctx["hdf"]
    n_ok = hctx["n_alignment_checked"]
    if hdf is None and trans_info["selective_reversible_input"]:
        raise ValueError(
            "含 loss 的 transitions 评估需要 feature_hierarchy.csv 来构造 selective loss mask; "
            "请把文件放回数据集目录或用 --hierarchy 指定。"
        )

    if args.no_meta:
        requested_fields_for_report = []
        requested_fields_source = "--no-meta"
    elif requested_meta_fields is None:
        requested_fields_for_report = list(TDL.META_FIELDS)
        requested_fields_source = "default_all"
    else:
        requested_fields_for_report = list(requested_meta_fields)
        requested_fields_source = "--meta-fields"

    feat_meta = None
    meta_info = {
        "used": False,
        "reason": None,
        "fields": [],
        "requested_fields": requested_fields_for_report,
        "requested_fields_source": requested_fields_source,
        "hierarchy": (None if hpath is None else os.path.abspath(hpath)),
        "enabled": {},
        "skipped": {},
        "n_alignment_checked": int(n_ok),
    }
    if args.no_meta:
        meta_info["reason"] = "disabled by --no-meta"
    elif args.dl_arch != "transformer":
        meta_info["reason"] = f"arch={args.dl_arch} 不使用元数据 (仅 transformer 生效)"
    elif hdf is None:
        meta_info["reason"] = "未找到 feature_hierarchy.csv, 退化为纯身份+状态"
        print(f"[phylo_eval][dl][warn] {meta_info['reason']}")
    else:
        feat_meta, _, meta_scan = TDL.load_feature_meta(
            hdf, requested_meta_fields=requested_meta_fields)
        meta_info.update(enabled=meta_scan["enabled"], skipped=meta_scan["skipped"])
        if feat_meta is None:
            meta_info["reason"] = "hierarchy 存在, 但所请求元数据字段在当前数据里都无区分度"
            print(f"[phylo_eval][dl][warn] {meta_info['reason']}")
        else:
            meta_info.update(
                used=True,
                fields=feat_meta["fields"],
                vocab={k: int(v) for k, v in feat_meta["vocab"].items()},
            )
            print(f"[phylo_eval][dl] 方案A 元数据已接入: fields={feat_meta['fields']} vocab={meta_info['vocab']}")
            if meta_info["skipped"]:
                print(f"[phylo_eval][dl] 自动跳过无信息量字段: {meta_info['skipped']}")

    head_mode = "gain_only"
    loss_info = {
        "requested": bool(trans_info["selective_reversible_input"]),
        "used": False,
        "reason": None,
        "source": None,
        "hierarchy": hpath,
        "n_alignment_checked": int(n_ok),
        "n_loss_total": int(trans_info["n_loss_total"]),
        "n_loss_transitions": int(trans_info["n_loss_transitions"]),
        "n_loss_eligible_features": 0,
        "loss_eligible_features": [],
    }
    if loss_scan:
        loss_info.update({k: v for k, v in loss_scan.items() if k != "hierarchy"})
    loss_info["head_mode_arg"] = args.dl_head_mode
    if args.dl_head_mode == "gain_only":
        loss_info["used"] = False
        loss_info["reason"] = "强制 gain-only (--dl-head-mode gain_only): 双头消融对照"
        print(f"[phylo_eval][dl] {loss_info['reason']}")
        return {
            "feat_meta": feat_meta,
            "meta_info": meta_info,
            "loss_mask": None,
            "loss_info": loss_info,
            "head_mode": "gain_only",
        }
    if trans_info["selective_reversible_input"]:
        if args.dl_arch != "transformer":
            raise ValueError(
                "gain+loss dual-head 当前仅支持 --dl-arch transformer; "
                "如需在其它 arch 上跑含 loss 的数据, 请显式加 --dl-head-mode gain_only")
        if trans_info["n_loss_total"] > 0 and (loss_mask is None or int(loss_mask.sum()) <= 0):
            raise ValueError(
                "transition 中检测到 loss, 但 hierarchy 未找到可逆先验; "
                "请检查 transitions 文件与 feature_hierarchy.csv 是否匹配 "
                "(loss 侧候选集合无法界定)。"
            )
        if loss_mask is not None and int(loss_mask.sum()) > 0:
            head_mode = "dual_head_selective_reversible"
            loss_info["used"] = True
            print(
                "[phylo_eval][dl] gain+loss dual-head 已启用: "
                f"source={loss_info['source']} n={loss_info['n_loss_eligible_features']}"
            )
        else:
            loss_info["reason"] = "含 loss 的输入, 但 hierarchy 中未找到可逆先验 -> 退化为 gain-only"
            print(f"[phylo_eval][dl][warn] {loss_info['reason']}")
    else:
        loss_info["reason"] = "irreversible/gain-only input"
        loss_mask = None

    return {
        "feat_meta": feat_meta,
        "meta_info": meta_info,
        "loss_mask": loss_mask,
        "loss_info": loss_info,
        "head_mode": head_mode,
    }


def _gain_only_dest_state(t: dict, L: int) -> np.ndarray:
    src = np.asarray(t["src"], dtype=int)
    dst = src.copy()
    for g in t.get("gained", []):
        gi = int(g)
        if 0 <= gi < L:
            dst[gi] = 1
    return dst


def _read_log_tail(path: str, n_lines: int = 20) -> str:
    if not path or not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = [ln.strip() for ln in f.readlines()[-n_lines:] if ln.strip()]
        return " | ".join(lines)
    except OSError:
        return ""


def resolve_hypertraps_runtime(root: Path, ht_bin: str | None = None,
                               ht_post_process: str | None = None) -> Dict[str, Any]:
    search_roots = [root.resolve()]
    parent_root = root.resolve().parent
    if parent_root not in search_roots:
        search_roots.append(parent_root)

    if ht_bin:
        candidates = [(Path(ht_bin).expanduser().resolve(), ht_post_process)]
    else:
        candidates = []
        for base in search_roots:
            candidates.extend([
                ((base / "refs" / "HyperTraPS" / "bin" / "RUN_MCMC_SAMPLER").resolve(), None),
                ((base / "tools" / "HyperTraPS" / "bin" / "main_mcmc").resolve(), ht_post_process),
            ])

    issues: List[str] = []
    for bin_path, post_hint in candidates:
        if not bin_path.exists():
            issues.append(f"missing binary: {bin_path}")
            continue
        if not os.access(bin_path, os.X_OK):
            issues.append(f"binary not executable: {bin_path}")
            continue

        if bin_path.name == "RUN_MCMC_SAMPLER":
            return {
                "backend": "legacy_sampler",
                "bin": str(bin_path),
                "post_process": None,
            }

        if bin_path.name == "main_mcmc":
            post_path = (
                Path(post_hint).expanduser().resolve()
                if post_hint else
                (root / "tools" / "HyperTraPS" / "scripts" / "post_process.py").resolve()
            )
            if not post_path.exists():
                issues.append(f"missing post_process.py for {bin_path}: {post_path}")
                continue
            return {
                "backend": "main_mcmc",
                "bin": str(bin_path),
                "post_process": str(post_path),
            }

        issues.append(
            f"unsupported HyperTraPS binary name: {bin_path.name}"
            " (expected RUN_MCMC_SAMPLER or main_mcmc)"
        )

    detail = "; ".join(issues) if issues else "no runtime candidates checked"
    raise RuntimeError(
        "No usable HyperTraPS runtime found. "
        "Expected either refs/HyperTraPS/bin/RUN_MCMC_SAMPLER "
        "(compatible with step3_train.sh) or tools/HyperTraPS/bin/main_mcmc. "
        f"Details: {detail}"
    )


def _load_hypertraps_theta_predictor(L: int, forwards_path: str):
    ht = HyperTraPSThetaBaseline(L, forwards_path, name="HyperTraPSTheta")
    ht.samples = [F for F in ht.samples if np.abs(F).sum() > 1e-9] or ht.samples
    if not ht.samples:
        raise RuntimeError(f"HyperTraPS produced empty posterior samples: {forwards_path}")
    return ht.predict_fn, len(ht.samples)


def _run_legacy_hypertraps(train_trans: List[dict], L: int, runtime: Dict[str, Any],
                           model="second-order", N=200, burn=5000, r=100,
                           timeout_sec=1800, threads=2):
    tmp = tempfile.mkdtemp(prefix="ht_fold_")
    log_path = os.path.join(tmp, "ht_run.log")
    try:
        inp = os.path.join(tmp, "transitions.txt")
        with open(inp, "w", encoding="utf-8") as f:
            for t in train_trans:
                src = np.asarray(t["src"], dtype=int)
                dst_gain_only = _gain_only_dest_state(t, L)
                f.write(" ".join(str(int(x)) for x in src.tolist()) + "\n")
                f.write(" ".join(str(int(x)) for x in dst_gain_only.tolist()) + "\n")

        cmd = [
            runtime["bin"], "-f", "transitions.txt", "-M", model,
            "-N", str(N), "-r", str(r), "-b", str(burn),
            "-i", "100", "-S", "1", "-t", str(threads),
        ]
        print("  [ht] backend=legacy_sampler run:", " ".join(cmd))
        with open(log_path, "w", encoding="utf-8") as logf:
            subprocess.run(
                cmd,
                cwd=tmp,
                check=True,
                timeout=timeout_sec,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

        forwards_path = os.path.join(tmp, "forwards.txt")
        if not os.path.exists(forwards_path) or os.path.getsize(forwards_path) <= 0:
            tail = _read_log_tail(log_path)
            raise RuntimeError(
                "legacy RUN_MCMC_SAMPLER finished without forwards.txt"
                + (f"; log tail: {tail}" if tail else "")
            )

        predict_fn, n_posterior = _load_hypertraps_theta_predictor(L, forwards_path)
        print(f"  [ht] backend=legacy_sampler posterior_samples={n_posterior}")
        return predict_fn
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        tail = _read_log_tail(log_path)
        raise RuntimeError(
            f"legacy RUN_MCMC_SAMPLER failed ({type(exc).__name__})"
            + (f"; log tail: {tail}" if tail else "")
        ) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _run_modern_hypertraps(train_trans: List[dict], L: int, runtime: Dict[str, Any],
                           model="second-order", N=200, burn=5000, r=100,
                           timeout_sec=1800, threads=2):
    tmp = tempfile.mkdtemp(prefix="ht_fold_")
    run_log_path = os.path.join(tmp, "ht_run.log")
    post_log_path = os.path.join(tmp, "ht_post.log")
    try:
        import csv
        import pandas as pd

        inp = os.path.join(tmp, "transitions.txt")
        with open(inp, "w", newline="") as f:
            w = csv.writer(f, delimiter=" ")
            for t in train_trans:
                src = np.asarray(t["src"], dtype=int)
                dst_gain_only = _gain_only_dest_state(t, L)
                w.writerow(list(src) + list(dst_gain_only))

        outstem = os.path.join(tmp, "run")
        model_flag = {"zero-order": "zero", "first-order": "on", "second-order": "two"}[model]
        cmd = [runtime["bin"], "-m", model_flag, "-N", str(N), "-b", str(burn), "-i", inp,
               "-o", outstem, "-r", str(r), "-t", str(threads)]
        print("  [ht] backend=main_mcmc run:", " ".join(cmd))
        with open(run_log_path, "w", encoding="utf-8") as logf:
            subprocess.run(
                cmd,
                check=True,
                timeout=timeout_sec,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )

        csv_path = outstem + f"_supplementary_data_{r}_list-pord-trajectory.csv"
        if not os.path.exists(csv_path):
            outdir = outstem + "_supplementary/"
            cmd2 = [
                sys.executable,
                runtime["post_process"],
                "--i", outstem,
                "--b", "0",
                "--f", outdir,
                "--m", model_flag,
            ]
            with open(post_log_path, "w", encoding="utf-8") as logf:
                subprocess.run(
                    cmd2,
                    check=True,
                    timeout=timeout_sec,
                    stdout=logf,
                    stderr=subprocess.STDOUT,
                )
            csv_path = os.path.join(outdir, f"list-pord-trajectory-{r}.csv")
        if not os.path.exists(csv_path):
            tail = _read_log_tail(post_log_path) or _read_log_tail(run_log_path)
            raise RuntimeError(
                "HyperTraPS parameter CSV not found after post_process"
                + (f"; log tail: {tail}" if tail else "")
            )

        param = pd.read_csv(csv_path)
        x = np.asarray(param.iloc[:, 0].values, dtype=float)

        def predict_fn(src):
            src = np.asarray(src, dtype=int)
            if len(x) == L:
                scores = np.exp(x)
            elif len(x) == L + L * L:
                main = x[:L]
                inter = x[L:].reshape(L, L)
                scores = np.exp(main + src @ inter)
            elif len(x) == 1 + L + L * (L - 1):
                off0 = x[0]
                main = x[1:1 + L]
                inter_flat = x[1 + L:]
                inter = np.zeros((L, L), dtype=float)
                idx = 0
                for j in range(L):
                    for k in range(L):
                        if j == k:
                            continue
                        inter[j, k] = inter_flat[idx]
                        idx += 1
                scores = np.exp(off0 + main + src @ inter)
            else:
                scores = np.ones(L, dtype=float)
            scores = np.where(src == 0, scores, 0.0)
            if scores.sum() <= 0:
                scores = np.where(src == 0, 1.0, 0.0)
            return scores

        return predict_fn
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as exc:
        tail = _read_log_tail(post_log_path) or _read_log_tail(run_log_path)
        raise RuntimeError(
            f"main_mcmc backend failed ({type(exc).__name__})"
            + (f"; log tail: {tail}" if tail else "")
        ) from exc
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def build_and_run_hypertraps_predictor(train_trans: List[dict], L: int, runtime: Dict[str, Any],
                                       model="second-order", N=200, burn=5000, r=100,
                                       timeout_sec=1800, threads=2):
    if runtime["backend"] == "legacy_sampler":
        return _run_legacy_hypertraps(
            train_trans,
            L,
            runtime,
            model=model,
            N=N,
            burn=burn,
            r=r,
            timeout_sec=timeout_sec,
            threads=threads,
        )
    if runtime["backend"] == "main_mcmc":
        return _run_modern_hypertraps(
            train_trans,
            L,
            runtime,
            model=model,
            N=N,
            burn=burn,
            r=r,
            timeout_sec=timeout_sec,
            threads=threads,
        )
    raise RuntimeError(f"Unsupported HyperTraPS backend: {runtime['backend']}")


def main():
    ap = argparse.ArgumentParser(description="Phylogeny-grouped CV evaluator")
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-mode", default="grouped", choices=["grouped", "random"],
                    help="fold 划分口径: grouped=谱系整组进出, random=随机 K 折 (泄漏对照)")
    ap.add_argument("--hierarchy", default=None,
                    help="feature_hierarchy.csv 路径 (默认自动定位 transitions 同数据集下)")
    ap.add_argument("--no-meta", action="store_true",
                    help="关闭 transformer 的特征元数据 token 化")
    ap.add_argument("--meta-fields", default="",
                    help="仅启用指定 metadata 字段, 逗号分隔: layer,drug_class,mechanism,location,reversible")
    ap.add_argument("--eval-sides", default="auto", choices=["auto", "gain", "both"],
                    help=(
                        "评估通道 (方案B 双通道): gain=仅 gain 侧 (历史口径); "
                        "both=gain + loss 双侧分别归一化并分别报告; "
                        "auto=输入含 loss 且 hierarchy 提供可逆先验时自动启用 both, 否则退化为 gain。"
                        "只控制评估, 不控制模型 head (见 --dl-head-mode)"
                    ))
    ap.add_argument("--stratify-by", default="auto", choices=["auto", "none", "group_prefix"],
                    help=(
                        "评估分层 (不改训练、不进模型, 仅拆分测试集分别报告): "
                        "group_prefix=按 group_id 的 ':' 前缀分层 "
                        "(ecoli2026 Route-B 前缀即抗生素 MEC/NIT/TMP); "
                        "auto=所有 group_id 含 ':' 且前缀数在 [2,10] 时自动启用; none=关闭"
                    ))
    ap.add_argument("--dl-head-mode", default="auto", choices=["auto", "gain_only"],
                    help=(
                        "DL 模型 head 口径: auto=输入含 loss 且有可逆先验时启用 gain+loss 双头; "
                        "gain_only=强制只训 gain head (双头消融对照)。"
                        "与 --eval-sides 解耦, 便于单独考察'加 loss head 是否损害 gain 侧'"
                    ))

    ap.add_argument("--no-hypertraps", action="store_true")
    ap.add_argument("--ht-bin", default="",
                    help="显式指定 HyperTraPS 可执行文件；支持 refs/HyperTraPS/bin/RUN_MCMC_SAMPLER 与 tools/HyperTraPS/bin/main_mcmc")
    ap.add_argument("--ht-post-process", default="",
                    help="main_mcmc 后端的 post_process.py 路径；留空则自动定位 tools/HyperTraPS/scripts/post_process.py")
    ap.add_argument("--ht-model", default="second-order", choices=["zero-order", "first-order", "second-order"])
    ap.add_argument("--ht-N", type=int, default=200)
    ap.add_argument("--ht-burn", type=int, default=5000)
    ap.add_argument("--ht-r", type=int, default=100)
    ap.add_argument("--ht-timeout", type=int, default=1800)
    ap.add_argument("--ht-threads", type=int, default=2)

    ap.add_argument("--dl-arch", default="", choices=["", "mlp", "transformer"])
    ap.add_argument("--dl-epochs", type=int, default=200)
    ap.add_argument("--dl-d-model", type=int, default=64)
    ap.add_argument("--dl-n-layers", type=int, default=2)
    ap.add_argument("--dl-n-heads", type=int, default=4)
    ap.add_argument("--dl-dropout", type=float, default=0.3)
    ap.add_argument("--dl-weight-decay", type=float, default=1e-3)
    ap.add_argument("--dl-lr", type=float, default=1e-3)
    ap.add_argument("--dl-batch-size", type=int, default=64)
    ap.add_argument("--dl-val-frac", type=float, default=0.15)
    ap.add_argument("--dl-patience", type=int, default=20)

    args = ap.parse_args()
    requested_meta_fields = TDL.parse_meta_fields_arg(args.meta_fields)
    if args.no_meta and requested_meta_fields is not None:
        raise SystemExit("--no-meta 与 --meta-fields 不能同时使用")

    os.makedirs(args.outdir, exist_ok=True)

    print(f">>> load transitions: {args.transitions}")
    trans_all, L, trans_info = TDL.load_transitions(args.transitions)

    # ---- hierarchy / loss 候选掩码: 一次加载, evaluator 与 DL 共用同一份先验 ----
    # dl_loss_mask 只由数据与 hierarchy 决定, 不受 --eval-sides 影响, 保证
    # "评估通道" 与 "模型 head" 两件事解耦。
    hctx = load_hierarchy_context(args, args.transitions, trans_all, L)
    dl_loss_mask, loss_mask_info = resolve_loss_eval_mask(hctx, trans_info)
    loss_mask = dl_loss_mask

    # ---- 决定启用哪些评估通道 ----
    n_loss_edges = sum(1 for t in trans_all if len(t["lost"]) > 0)
    loss_side_available = (loss_mask is not None) and n_loss_edges > 0
    if args.eval_sides == "gain":
        sides = [GAIN]
        loss_mask_info["eval_side_decision"] = "disabled by --eval-sides gain"
    elif args.eval_sides == "both":
        if not loss_side_available:
            raise SystemExit(
                "--eval-sides both 需要: (1) transitions 中存在 loss 边, "
                "(2) feature_hierarchy.csv 提供可逆先验以界定 loss 候选。"
                f" 当前 n_loss_edges={n_loss_edges}, loss_mask={loss_mask_info.get('reason')}"
            )
        sides = [GAIN, LOSS]
        loss_mask_info["eval_side_decision"] = "enabled by --eval-sides both"
    else:  # auto
        sides = [GAIN, LOSS] if loss_side_available else [GAIN]
        loss_mask_info["eval_side_decision"] = (
            "auto -> gain+loss" if loss_side_available
            else f"auto -> gain only ({loss_mask_info.get('reason') or 'no loss edges'})"
        )
    if LOSS not in sides:
        loss_mask = None
    loss_mask_info["eval_sides"] = list(sides)
    print(f">>> eval sides = {sides} ({loss_mask_info['eval_side_decision']})")
    if LOSS in sides:
        print(
            f">>> loss 候选先验: source={loss_mask_info['source']} "
            f"n_loss_eligible_features={loss_mask_info['n_loss_eligible_features']}"
        )

    # ---- 评估边集合: 两侧共用同一套 fold 划分, 取两侧评估边的并集 ----
    # sides==[gain] 时该并集恰等于历史的 gained>0 集合, 结果与旧版逐位一致。
    def _is_eval_edge(t):
        if len(t["gained"]) > 0:
            return True
        return LOSS in sides and len(t["lost"]) > 0

    trans_eval = [t for t in trans_all if _is_eval_edge(t)]
    if not trans_eval:
        raise SystemExit(
            "当前 transitions 文件中没有可用于评估的边 "
            f"(sides={sides}, n_gained=0 且 n_lost=0 或 loss 侧未启用)")

    profile = transition_profile(trans_all, trans_eval, sides=sides)
    stratify_mode, stratify_info = resolve_stratify_mode(trans_eval, args.stratify_by)
    print(f">>> stratify_by = {stratify_mode} ({stratify_info['reason']}) "
          f"labels={stratify_info['labels']}")
    groups = [t["group"] for t in trans_eval]
    if args.split_mode == "grouped":
        folds = group_kfold_indices(groups, n_splits=args.folds, seed=args.seed)
        if not folds:
            raise SystemExit("无法构造有效的 group K-fold 划分")
    else:
        folds = random_kfold_indices(len(trans_eval), n_splits=args.folds, seed=args.seed)
        if not folds:
            raise SystemExit("无法构造有效的 random K-fold 划分")

    print(
        f">>> input={len(trans_all)} eval={len(trans_eval)} features={L} "
        f"split_mode={args.split_mode} groups={len(set(groups))} folds={len(folds)} "
        f"gain_edges={profile['n_gain_side_edges']} loss_edges={profile['n_loss_side_edges']} "
        f"skipped_non_gain={profile['n_skipped_non_gain']}"
    )
    print(
        f">>> edge profile: single_gain={profile['single_gain']} "
        f"multi_gain={profile['multi_gain']} mixed={profile['mixed_gain_loss']} "
        f"loss_only={profile['loss_only']} single_loss={profile['single_loss']} "
        f"multi_loss={profile['multi_loss']}"
    )

    ht_runtime = None
    ht_runtime_error = None
    if not args.no_hypertraps:
        try:
            ht_runtime = resolve_hypertraps_runtime(
                Path(ROOT),
                args.ht_bin or None,
                args.ht_post_process or None,
            )
            msg = f"[phylo_eval][ht] backend={ht_runtime['backend']} bin={ht_runtime['bin']}"
            if ht_runtime.get("post_process"):
                msg += f" post_process={ht_runtime['post_process']}"
            print(msg)
        except Exception as exc:
            ht_runtime_error = str(exc)
            print(f"[phylo_eval][ht][warn] {ht_runtime_error}")

    dl_setup = None
    if args.dl_arch:
        dl_setup = prepare_dl_setup(
            args, hctx, trans_info, requested_meta_fields,
            loss_mask=dl_loss_mask, loss_scan=loss_mask_info)

    method_order = ["B0_uniform", "B1_prevalence"]
    dl_label = f"DL_{args.dl_arch}" if args.dl_arch else None
    ht_label = f"HT_{args.ht_model}" if not args.no_hypertraps else None
    if dl_label:
        method_order.append(dl_label)
    if ht_label:
        method_order.append(ht_label)
    # loss 侧参与评估的方法 (HyperTraPS 原理上只支持单调累积, 排除)
    loss_method_order = [
        m for m in method_order
        if not m.startswith(LOSS_UNSUPPORTED_PREFIXES)
    ]
    # 综合通道: 统一事件空间上的单一性能。gain-only 方法 (含 HT) 也可参与 ——
    # pi<1 时它们在 loss 侧退化为该侧 uniform, 相当于"不会预测 loss"的诚实惩罚。
    combined_method_order = list(method_order)

    folds_payload = []
    per_item = {m: [] for m in method_order}
    per_item_loss = {m: [] for m in loss_method_order}
    per_item_combined = {m: [] for m in combined_method_order}
    dl_fit_reports = []
    fold_runtimes: List[Dict[str, Any]] = []
    compute_env = describe_compute_environment()
    print(
        "[runtime] device: dl={} cuda_available={} cpu_count={} torch_threads={} ht_threads={}".format(
            compute_env["dl_device"],
            compute_env["cuda_available"],
            compute_env["cpu_count"],
            compute_env["torch_num_threads"],
            args.ht_threads,
        )
    )

    for fi, (tr_idx, te_idx) in enumerate(folds):
        train = [trans_eval[i] for i in tr_idx]
        test = [trans_eval[i] for i in te_idx]
        print(f"\n=== Fold {fi + 1}/{len(folds)}: train={len(train)} test={len(test)} ===")

        runtime_row: Dict[str, Any] = {"fold": fi, "methods": {}}

        b0, t_b0 = _timed(lambda: UniformBaseline(L).fit(train))
        b1, t_b1 = _timed(lambda: PrevalenceBaseline(L).fit(train))
        runtime_row["methods"]["B0_uniform"] = {"fit_sec": t_b0}
        runtime_row["methods"]["B1_prevalence"] = {"fit_sec": t_b1}
        predictors = {
            "B0_uniform": b0.predict_fn,
            "B1_prevalence": b1.predict_fn,
        }
        # loss 侧预测器: B0 与 gain 侧共用 (全 1, 由 evaluator 在 loss 候选上归一);
        # B1 换成"被丢失特征"频次, 保持与 gain 侧同构的边际频率对照。
        predictors_loss: Dict[str, Any] = {}
        if LOSS in sides:
            b1l, t_b1l = _timed(lambda: LossPrevalenceBaseline(L).fit(train))
            predictors_loss["B0_uniform"] = b0.predict_fn
            predictors_loss["B1_prevalence"] = b1l.predict_fn
            runtime_row["methods"]["B1_prevalence"]["fit_sec_loss"] = t_b1l

        if dl_label:
            def _fit_dl():
                return MDL.SetPredDL(
                    L,
                    arch=args.dl_arch,
                    d_model=args.dl_d_model,
                    dropout=args.dl_dropout,
                    weight_decay=args.dl_weight_decay,
                    epochs=args.dl_epochs,
                    seed=args.seed + fi,
                    verbose=False,
                    feat_meta=dl_setup["feat_meta"],
                    head_mode=dl_setup["head_mode"],
                    loss_mask=dl_setup["loss_mask"],
                ).fit(train)

            dl, t_dl = _timed(_fit_dl)
            runtime_row["methods"][dl_label] = {"fit_sec": t_dl}
            predictors[dl_label] = dl.predict_fn
            if LOSS in sides and dl.fit_report.get("loss_head_enabled"):
                predictors_loss[dl_label] = dl.predict_loss_fn
            rep = dict(dl.fit_report)
            rep.update({
                "fold": fi,
                "arch": args.dl_arch,
                "d_model": args.dl_d_model,
                "dropout": args.dl_dropout,
                "weight_decay": args.dl_weight_decay,
                "epochs": args.dl_epochs,
                "fit_sec": t_dl,
                "device": compute_env["dl_device"],
            })
            dl_fit_reports.append(rep)
            print(f"  [dl] fold {fi} fit ({t_dl:.1f}s): {rep}")

        if ht_label:
            if ht_runtime is None:
                print(f"  [ht] WARN: {ht_runtime_error}; skip HT on this fold")
                runtime_row["methods"][ht_label] = {
                    "fit_sec": None,
                    "status": "unavailable",
                    "reason": ht_runtime_error,
                }
            else:
                # HyperTraPS 只支持单调累积: 只喂含 gain 的边, 否则 loss-only 边会
                # 退化成 src==dst 的自环样本, 污染 MCMC。
                train_ht = [t for t in train if len(t["gained"]) > 0]
                t0 = time.perf_counter()
                try:
                    predictors[ht_label] = build_and_run_hypertraps_predictor(
                        train_ht,
                        L,
                        runtime=ht_runtime,
                        model=args.ht_model,
                        N=args.ht_N,
                        burn=args.ht_burn,
                        r=args.ht_r,
                        timeout_sec=args.ht_timeout,
                        threads=args.ht_threads,
                    )
                    t_ht = time.perf_counter() - t0
                    runtime_row["methods"][ht_label] = {
                        "fit_sec": t_ht,
                        "status": "ok",
                        "threads": int(args.ht_threads),
                        "timeout_sec": float(args.ht_timeout),
                    }
                    print(f"  [ht] fold {fi} done in {t_ht:.1f}s")
                except Exception as exc:
                    t_ht = time.perf_counter() - t0
                    runtime_row["methods"][ht_label] = {
                        "fit_sec": t_ht,
                        "status": "failed",
                        "reason": str(exc),
                        "threads": int(args.ht_threads),
                        "timeout_sec": float(args.ht_timeout),
                    }
                    print(
                        f"  [ht] WARN: HyperTraPS fold {fi} failed after {t_ht:.1f}s: {exc}; "
                        "skip HT on this fold"
                    )

        n_test_gain = sum(1 for t in test if len(t["gained"]) > 0)
        n_test_loss = sum(1 for t in test if len(t["lost"]) > 0)
        fold_row = {
            "fold": fi,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_train_gain": int(sum(1 for t in train if len(t["gained"]) > 0)),
            "n_train_loss": int(sum(1 for t in train if len(t["lost"]) > 0)),
            "n_test_gain": int(n_test_gain),
            "n_test_loss": int(n_test_loss),
            "methods": {},
        }
        for method in method_order:
            if method not in predictors:
                continue
            ev, t_pred = _timed(eval_transitions, test, predictors[method], side=GAIN)
            sm = summarize(ev)
            fold_row["methods"][method] = _jsonable(sm)
            runtime_row["methods"].setdefault(method, {})["predict_sec"] = t_pred
            per_item[method].append(np.asarray(ev["per_item_logl"], dtype=float))
        attach_logl_lift(fold_row["methods"])
        for method in method_order:
            if method in fold_row["methods"]:
                print("  " + fmt_metrics(method, fold_row["methods"][method], side=GAIN))

        if LOSS in sides:
            fold_row["methods_loss"] = {}
            for method in loss_method_order:
                if method not in predictors_loss:
                    continue
                ev_l, t_pred_l = _timed(
                    eval_transitions, test, predictors_loss[method],
                    side=LOSS, loss_mask=loss_mask)
                sm_l = summarize(ev_l)
                fold_row["methods_loss"][method] = _jsonable(sm_l)
                runtime_row["methods"].setdefault(method, {})["predict_sec_loss"] = t_pred_l
                per_item_loss[method].append(np.asarray(ev_l["per_item_logl"], dtype=float))
            attach_logl_lift(fold_row["methods_loss"])
            for method in loss_method_order:
                if method in fold_row["methods_loss"]:
                    print("  " + fmt_metrics(method, fold_row["methods_loss"][method], side=LOSS))

            # ---- 综合通道: 统一事件空间, 单一综合性能 ----
            # pi 只从训练折估计, 避免测试折信息泄漏
            pi = estimate_gain_event_share(train)
            fold_row["pi_gain"] = float(pi)
            fold_row["methods_combined"] = {}
            for method in combined_method_order:
                if method not in predictors:
                    continue
                # 该方法若无 loss 侧预测能力 (如 HT / 单头 DL), 用该侧 uniform 兜底:
                # 不额外惩罚也不额外奖励, 由综合指标自然体现其缺失
                lfn = predictors_loss.get(method, b0.predict_fn)
                ev_c = eval_transitions_combined(
                    test, predictors[method], loss_predict_fn=lfn,
                    loss_mask=loss_mask, pi=pi)
                sm_c = summarize(ev_c)
                fold_row["methods_combined"][method] = _jsonable(sm_c)
                per_item_combined[method].append(
                    np.asarray(ev_c["per_item_logl"], dtype=float))
            attach_logl_lift(fold_row["methods_combined"])
            for method in combined_method_order:
                if method in fold_row["methods_combined"]:
                    print("  " + fmt_metrics(
                        method, fold_row["methods_combined"][method], side=COMBINED))

        # ---- 分层评估: 同一池化模型, 仅拆分测试集分别报告 ----
        if stratify_mode != "none":
            fold_row["strata"] = eval_fold_strata(
                test, train, stratify_mode, predictors, predictors_loss,
                method_order, loss_method_order, combined_method_order,
                sides, loss_mask, b0.predict_fn,
                pooled_pi=fold_row.get("pi_gain"))

        folds_payload.append(fold_row)
        fold_runtimes.append(runtime_row)

    metrics_nested, overall_by_method = aggregate_fold_metrics(folds_payload, method_order)
    metrics_loss_nested: Dict[str, Dict[str, float]] = {}
    overall_loss_by_method: Dict[str, Dict[str, float]] = {}
    metrics_combined_nested: Dict[str, Dict[str, float]] = {}
    overall_combined_by_method: Dict[str, Dict[str, float]] = {}
    if LOSS in sides:
        metrics_loss_nested, overall_loss_by_method = aggregate_fold_metrics(
            folds_payload, loss_method_order, methods_key="methods_loss")
        metrics_combined_nested, overall_combined_by_method = aggregate_fold_metrics(
            folds_payload, combined_method_order, methods_key="methods_combined")

    dl_mean_train_val_gap = None
    if dl_fit_reports:
        dl_mean_train_val_gap = _nanmean([rep.get("train_val_gap", float("nan")) for rep in dl_fit_reports])

    dl_vs_prevalence = compare_methods(per_item, dl_label, "B1_prevalence", "DL", "Prevalence", args.seed) if dl_label else None
    ht_vs_prevalence = compare_methods(per_item, ht_label, "B1_prevalence", "HyperTraPS", "Prevalence", args.seed + 1) if ht_label else None
    dl_vs_hypertraps = compare_methods(per_item, dl_label, ht_label, "DL", "HyperTraPS", args.seed + 2) if (dl_label and ht_label) else None

    comparisons = {
        "dl_vs_prevalence": dl_vs_prevalence,
        "ht_vs_prevalence": ht_vs_prevalence,
        "dl_vs_hypertraps": dl_vs_hypertraps,
    }
    loss_comparisons = {}
    combined_comparisons = {}
    if LOSS in sides:
        loss_comparisons = {
            "dl_vs_prevalence_loss": (
                compare_methods(per_item_loss, dl_label, "B1_prevalence",
                                "DL", "LossPrevalence", args.seed + 3)
                if dl_label else None
            ),
        }
        combined_comparisons = {
            "dl_vs_prevalence_combined": (
                compare_methods(per_item_combined, dl_label, "B1_prevalence",
                                "DL", "Prevalence", args.seed + 4)
                if dl_label else None
            ),
            "dl_vs_uniform_combined": (
                compare_methods(per_item_combined, dl_label, "B0_uniform",
                                "DL", "Uniform", args.seed + 5)
                if dl_label else None
            ),
        }

    fold_metrics_payload = {
        "transitions": os.path.abspath(args.transitions),
        "split_mode": args.split_mode,
        "eval_sides": list(sides),
        "transition_profile": profile,
        "folds": folds_payload,
        "fold_runtimes": fold_runtimes,
        "compute_environment": compute_env,
    }
    with open(os.path.join(args.outdir, "fold_metrics.json"), "w") as f:
        json.dump(_jsonable(fold_metrics_payload), f, indent=2, ensure_ascii=False)

    summary = {
        "transitions": os.path.abspath(args.transitions),
        "transitions_path": os.path.abspath(args.transitions),
        "transition_profile": profile,
        "transition_input": trans_info,
        "split_mode": args.split_mode,
        "n_transitions": int(len(trans_all)),
        "n_eval_transitions": int(len(trans_eval)),
        "n_skipped_non_gain_transitions": int(len(trans_all) - len(trans_eval)),
        "L": int(L),
        "n_features": int(L),
        "n_groups": int(len(set(groups))),
        "n_groups_input": int(len({t['group'] for t in trans_all})),
        "folds": int(len(folds)),
        "n_folds": int(len(folds)),
        "ht_model": (None if args.no_hypertraps else args.ht_model),
        "ht_runtime": (None if args.no_hypertraps else {
            "requested_bin": (os.path.abspath(args.ht_bin) if args.ht_bin else None),
            "requested_post_process": (os.path.abspath(args.ht_post_process) if args.ht_post_process else None),
            "backend": (None if ht_runtime is None else ht_runtime["backend"]),
            "bin": (None if ht_runtime is None else ht_runtime["bin"]),
            "post_process": (None if ht_runtime is None else ht_runtime.get("post_process")),
            "resolution_error": ht_runtime_error,
        }),
        # 顶层 metrics/overall/methods 始终为 gain 侧, 与历史 gain-only 口径一致
        "metrics": metrics_nested,
        "overall": overall_by_method,
        "methods": [m for m in method_order if m in overall_by_method],
        # 方案B 双通道: loss 侧单列, 不污染主结论
        "eval_sides": list(sides),
        "eval_sides_arg": args.eval_sides,
        "loss_side_eval": {
            "enabled": bool(LOSS in sides),
            "candidate_mask": loss_mask_info,
            "methods": [m for m in loss_method_order if m in overall_loss_by_method],
            "excluded_methods": [
                m for m in method_order if m.startswith(LOSS_UNSUPPORTED_PREFIXES)
            ],
            "excluded_reason": (
                "HyperTraPS 为单调累积模型, 不支持 1->0 事件, 故不参与 loss 侧评估"
            ),
            "metrics": metrics_loss_nested,
            "overall": overall_loss_by_method,
            "semantics": (
                "loss 侧的 1->0 为端点净状态消失/回退, 不等于已确证的回复突变事件; "
                "候选集合限定为 src==1 且属 hierarchy 可逆先验的位。"
            ),
            **loss_comparisons,
        },
        # 综合通道: 统一事件空间上的单一性能, 是"双头模型整体表现"的 headline 指标
        "combined_eval": {
            "enabled": bool(LOSS in sides),
            "pi_gain_per_fold": [f.get("pi_gain") for f in folds_payload],
            "pi_gain_mean": _nanmean([
                f.get("pi_gain", float("nan")) for f in folds_payload]),
            "methods": [m for m in combined_method_order if m in overall_combined_by_method],
            "metrics": metrics_combined_nested,
            "overall": overall_combined_by_method,
            "definition": (
                "事件空间 = {(gain,i): s_i==0} ∪ {(loss,j): s_j==1 且 j 属可逆先验}; "
                "P(gain,i)=pi*P_g(i), P(loss,j)=(1-pi)*P_l(j), 两侧各自归一后按 side "
                "权重 pi 混合。pi 仅由训练折的 gain/loss 事件比估计, 不含测试折信息。"
                "无 loss 预测能力的方法 (HyperTraPS / 单头) 在 loss 侧以该侧 uniform 兜底。"
                "loss 候选为空或 pi==1 时本指标恒等于 gain 侧指标, 故 gain-only 数据集上"
                "与历史结果同口径。"
            ),
            **combined_comparisons,
        },
        # 分层评估: 池化训练 + 分层报告。用于暴露"不同选择压力下性能不同"这类结构差异
        "stratified_eval": {
            "enabled": bool(stratify_mode != "none"),
            "mode": stratify_mode,
            "info": stratify_info,
            "note": (
                "fold 划分与训练均为池化, 分层只发生在评估阶段; 分层变量 (如抗生素) "
                "不作为模型输入, 符合'环境协变量只用于切分/报告, 不进模型'的设计口径。"
                "combined 通道的 pi 优先按该层训练子集估计 (层间 gain/loss 平衡可能不同)。"
            ),
            "strata": (
                aggregate_strata(
                    folds_payload, stratify_info["labels"],
                    {"gain": method_order, "loss": loss_method_order,
                     "combined": combined_method_order})
                if stratify_mode != "none" else {}
            ),
        },
        "dl_fit_reports": dl_fit_reports,
        "dl_training_setup": (None if dl_setup is None else {
            "transition_input": trans_info,
            "feat_meta": dl_setup["meta_info"],
            "loss_head": dl_setup["loss_info"],
        }),
        "dl_mean_train_val_gap": dl_mean_train_val_gap,
        "compute_environment": compute_env,
        "runtime_profile": {
            "per_method": summarize_runtime(fold_runtimes),
            "per_fold": fold_runtimes,
            "ht_threads": (None if args.no_hypertraps else int(args.ht_threads)),
            "ht_timeout_sec": (None if args.no_hypertraps else float(args.ht_timeout)),
            "note": (
                "fit_sec 为每折训练/推断构建耗时, predict_sec 为该折测试集打分耗时; "
                "HyperTraPS 失败折的 fit_sec 记录的是失败前已消耗时间 (通常等于 timeout)。"
            ),
        },
        **comparisons,
    }
    with open(os.path.join(args.outdir, "eval_summary.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)

    print("\n=== Overall (gain side) ===")
    for method in summary["methods"]:
        print(fmt_metrics(method, overall_by_method[method], side=GAIN))

    if LOSS in sides:
        print("\n=== Overall (loss side) ===")
        for method in loss_method_order:
            if method in overall_loss_by_method:
                print(fmt_metrics(method, overall_loss_by_method[method], side=LOSS))
        print(f"\n=== Overall (combined, pi_gain={summary['combined_eval']['pi_gain_mean']:.3f}) ===")
        for method in combined_method_order:
            if method in overall_combined_by_method:
                print(fmt_metrics(method, overall_combined_by_method[method], side=COMBINED))

    if stratify_mode != "none":
        strata_payload = summary["stratified_eval"]["strata"]
        for label in stratify_info["labels"]:
            entry = strata_payload.get(label) or {}
            pis = [x for x in (entry.get("pi_gain_per_fold") or []) if x is not None]
            pi_txt = f" pi_gain={_nanmean(pis):.3f}" if pis else ""
            print(f"\n=== Stratum [{label}]{pi_txt} ===")
            for channel in (GAIN, LOSS, COMBINED):
                blk = (entry.get(channel) or {}).get("overall") or {}
                for method, vals in blk.items():
                    print("  " + fmt_metrics(f"{label}/{method}", vals, side=channel))

    rt = summary["runtime_profile"]["per_method"]
    if rt:
        print("\n=== Runtime (wall-clock) ===")
        print(
            "  device: dl={} cuda={} cpu_count={} torch_threads={} ht_threads={}".format(
                compute_env["dl_device"],
                compute_env["cuda_available"],
                compute_env["cpu_count"],
                compute_env["torch_num_threads"],
                (None if args.no_hypertraps else args.ht_threads),
            )
        )
        for method in method_order:
            if method not in rt:
                continue
            e = rt[method]
            fit_t = e["fit_sec_total"]
            pred_t = e["predict_sec_total"]
            print(
                "  {:<18} fit={:>9} s  predict={:>8} s  total={:>9} s  ({} folds)".format(
                    method,
                    ("n/a" if fit_t is None else f"{fit_t:.2f}"),
                    ("n/a" if pred_t is None else f"{pred_t:.2f}"),
                    ("n/a" if e["total_sec"] is None else f"{e['total_sec']:.2f}"),
                    e["n_folds_timed"],
                )
            )

    if dl_vs_prevalence:
        print("\n=== DL vs Prevalence ===")
        print(dl_vs_prevalence)
    if ht_vs_prevalence:
        print("\n=== HyperTraPS vs Prevalence ===")
        print(ht_vs_prevalence)
    if dl_vs_hypertraps:
        print("\n=== DL vs HyperTraPS ===")
        print(dl_vs_hypertraps)
    if loss_comparisons.get("dl_vs_prevalence_loss"):
        print("\n=== [loss] DL vs LossPrevalence ===")
        print(loss_comparisons["dl_vs_prevalence_loss"])

    print(f"\nSaved: {args.outdir}/eval_summary.json, {args.outdir}/fold_metrics.json")


if __name__ == "__main__":
    main()
