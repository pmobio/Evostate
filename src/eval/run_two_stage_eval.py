#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import sys
import time
import csv
from pathlib import Path
from typing import List

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS_DIR = os.path.join(ROOT, "models")
if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import train_dl as TDL
from baselines import LossPrevalenceBaseline, PrevalenceBaseline, UniformBaseline
from evaluator import (
    estimate_gain_event_share,
    eval_transitions,
    eval_transitions_combined,
    group_kfold_indices,
    random_kfold_indices,
    summarize,
)
from run_phylo_eval import (
    COMBINED,
    GAIN,
    LOSS,
    LOSS_UNSUPPORTED_PREFIXES,
    _jsonable,
    _nanmean,
    aggregate_fold_metrics,
    aggregate_strata,
    attach_logl_lift,
    eval_fold_strata,
    resolve_stratify_mode,
    build_and_run_hypertraps_predictor,
    compare_methods,
    fmt_metrics,
    resolve_hypertraps_runtime,
    transition_profile,
)


def _load_two_stage_backbone_module():
    module_path = os.path.join(MODELS_DIR, "two_stage_backbone.py")
    module_name = "two_stage_backbone_runtime"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 two_stage_backbone 模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TSB = _load_two_stage_backbone_module()
TwoStageTransitionModel = _TSB.TwoStageTransitionModel
build_state_examples = _TSB.build_state_examples
filter_gain_transitions = _TSB.filter_gain_transitions
filter_event_transitions = _TSB.filter_event_transitions
prepare_feature_setup = _TSB.prepare_feature_setup


def _nanmean(xs):
    arr = np.asarray(xs, dtype=float)
    if arr.size == 0:
        return float("nan")
    valid = arr[~np.isnan(arr)]
    if valid.size == 0:
        return float("nan")
    return float(valid.mean())


def _compare_if_available(per_item, method_a, method_b, label_a, label_b, seed):
    if method_a not in per_item or method_b not in per_item:
        return None
    if not per_item[method_a] or not per_item[method_b]:
        return None
    return compare_methods(per_item, method_a, method_b, label_a, label_b, seed)


def dump_fold_predictions(path, test, predictors, L):
    """Persist the out-of-fold next-step distribution of every method on one fold.

    Purely additive: it re-applies each ``predict_fn`` to the same held-out transitions,
    with the same candidate mask and the same normalisation ``eval_transitions`` uses, and
    only writes the result out. No metric depends on it, so enabling the dump cannot change
    any number in eval_summary.json.

    The item filter mirrors ``eval_transitions`` exactly (non-empty truth, non-empty candidate
    set, truth inside the candidate set), so row i of the dump is item i of the evaluation.
    """
    from evaluator import _candidate_mask, _normalize_on_candidates

    keep = []
    for t in test:
        truth = list(t.get("gained", []) or [])
        if not truth:
            continue
        cand = np.flatnonzero(_candidate_mask(t["src"], side=GAIN))
        if len(cand) == 0 or not set(truth).issubset(set(cand.tolist())):
            continue
        keep.append(t)
    if not keep:
        return None

    n = len(keep)
    src = np.zeros((n, L), dtype=np.int8)
    truth_mask = np.zeros((n, L), dtype=np.int8)
    groups = []
    for i, t in enumerate(keep):
        src[i] = np.asarray(t["src"], dtype=np.int8)
        truth_mask[i, list(t.get("gained", []) or [])] = 1
        groups.append(str(t.get("group", "")))

    out = {"src": src, "truth": truth_mask, "group": np.asarray(groups, dtype=object)}
    for m, fn in predictors.items():
        P = np.zeros((n, L), dtype=np.float32)
        for i, t in enumerate(keep):
            P[i] = _normalize_on_candidates(fn(t["src"]), t["src"], side=GAIN)
        out["p__" + m] = P
    np.savez_compressed(path, **out)
    return path


def resolve_loss_setup(setup, args) -> tuple:
    """从 hierarchy 的可逆先验构造 loss 侧候选掩码 (evaluator 与 Stage 2 共用)。

    返回 (mask 或 None, info)。与 --eval-sides 解耦: mask 只由数据与 hierarchy
    决定, 评估通道的开关另行判断。
    """
    trans_info = setup.transition_info
    info = {
        "hierarchy": setup.hierarchy_path,
        "n_alignment_checked": setup.meta_info.get("n_alignment_checked", 0),
        "n_loss_total": int(trans_info["n_loss_total"]),
        "n_loss_transitions": int(trans_info["n_loss_transitions"]),
        "used": False,
        "reason": None,
        "source": None,
        "n_loss_eligible_features": 0,
        "loss_eligible_features": [],
    }
    if setup.hierarchy_df is None:
        info["reason"] = "未找到 feature_hierarchy.csv, 无法界定 loss 侧候选集合"
        return None, info
    mask, scan = TDL.load_selective_loss_mask(setup.hierarchy_df)
    info.update(scan)
    if int(mask.sum()) <= 0:
        info["reason"] = scan.get("reason") or "hierarchy 中未找到可逆先验"
        return None, info
    info["used"] = True
    info["reason"] = "ok"
    return mask, info


def _build_model(args, setup, seed, head_mode="gain_only", loss_mask=None):
    return TwoStageTransitionModel(
        head_mode=head_mode,
        loss_mask=loss_mask,
        loss_weight=args.loss_weight,
        L=setup.L,
        arch=args.arch,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        patience=args.patience,
        seed=seed,
        feat_meta=setup.feat_meta,
    )


def main():
    ap = argparse.ArgumentParser(description="Fold-wise two-stage evaluator")
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--dump-predictions", action="store_true",
                    help="额外写出每折的折外预测分布 (predictions/predictions_fold_XX.npz), "
                         "供 lineage 级分析使用; 不影响任何指标")
    ap.add_argument("--dump-fold-models", action="store_true",
                    help="额外写出每折的 stage2 checkpoint 与该折留出的 group_id "
                         "(fold_runs/fold_XX/{stage2_model.pt,holdout_groups.json}), "
                         "供折外条件预测分析对任意合成状态打分; 不影响任何指标")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split-mode", default="grouped", choices=["grouped", "random"])
    ap.add_argument("--hierarchy", default=None)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--meta-fields", default="")
    ap.add_argument("--eval-sides", default="auto", choices=["auto", "gain", "both"],
                    help=(
                        "评估通道 (方案B 双通道): gain=仅 gain 侧 (历史口径); "
                        "both=gain + loss 双侧各自归一化并分别报告; "
                        "auto=输入含 loss 且 hierarchy 提供可逆先验时自动启用 both。"
                        "只控制评估, 不控制模型 head (见 --ts-head-mode)"
                    ))
    ap.add_argument("--ts-head-mode", default="auto", choices=["auto", "gain_only"],
                    help=(
                        "Stage 2 head 口径: auto=输入含 loss 且有可逆先验时启用 gain+loss 双头; "
                        "gain_only=强制只训 gain head (双头消融对照)"
                    ))
    ap.add_argument("--loss-weight", type=float, default=1.0,
                    help="loss head 在 Stage 2 总目标中的权重 (gain 侧固定为 1.0)")
    ap.add_argument("--stratify-by", default="auto", choices=["auto", "none", "group_prefix"],
                    help=(
                        "评估分层 (不改训练、不进模型, 仅拆分测试集分别报告): "
                        "group_prefix=按 group_id 的 ':' 前缀分层 "
                        "(ecoli2026 Route-B 前缀即抗生素 MEC/NIT/TMP); auto=自动判定; none=关闭"
                    ))
    ap.set_defaults(use_hypertraps=False)
    ap.add_argument("--with-hypertraps", dest="use_hypertraps", action="store_true",
                    help="启用 HyperTraPS，并与 TS_pretrained 在同一 grouped-CV 口径下比较")
    ap.add_argument("--no-hypertraps", dest="use_hypertraps", action="store_false",
                    help="禁用 HyperTraPS（默认）")
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
    ap.add_argument("--arch", default="transformer", choices=["mlp", "transformer"])
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--pretrain-epochs", type=int, default=120)
    ap.add_argument("--finetune-epochs", type=int, default=120)
    ap.add_argument("--aligned-epochs", type=int, default=0,
                    help="在 stage1 与 stage2 之间执行的最小 alignment epoch 数；0 表示关闭")
    ap.add_argument("--aligned-lr-scale", type=float, default=0.5,
                    help="alignment 阶段相对 stage2 学习率缩放")
    ap.add_argument("--mask-prob", type=float, default=0.15)
    ap.add_argument("--mask-prob-pos", type=float, default=None)
    ap.add_argument("--mask-prob-neg", type=float, default=None)
    ap.add_argument("--contrastive-weight", type=float, default=0.0)
    ap.add_argument("--contrastive-temperature", type=float, default=0.1)
    ap.add_argument("--contrastive-warmup-epochs", type=int, default=0)
    ap.add_argument("--similar-state-neg-threshold", type=float, default=1.0)
    ap.add_argument(
        "--pretrain-checkpoint-metric",
        default="val_loss",
        choices=["val_loss", "val_recon_loss", "val_masked_accuracy"],
    )
    ap.add_argument("--state-source", default="src_dest", choices=["src", "dest", "src_dest", "all"])
    ap.add_argument("--no-dedupe-states", action="store_true")
    ap.add_argument("--skip-random-init-control", action="store_true",
                    help="仅训练 two-stage pretrained，不再额外训练 random-init 对照")
    ap.add_argument("--frozen-backbone-eval", action="store_true",
                    help="增加 frozen-backbone linear evaluation：冻结 backbone，仅训练 stage2 gain 线性头")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=0,
                    help="Freeze→Unfreeze 策略中，先冻结 backbone 训练的 epoch 数")
    ap.add_argument("--backbone-lr-scale", type=float, default=1.0,
                    help="Stage 2 中 backbone 学习率相对 base lr 的缩放")
    ap.add_argument("--head-lr-scale", type=float, default=1.0,
                    help="Stage 2 中 gain head 学习率相对 base lr 的缩放")
    ap.add_argument("--unfreeze-last-n-layers", type=int, default=0,
                    help="Partial unfreeze 时仅训练 transformer 最后 N 层；0 表示全 backbone")
    ap.add_argument("--freeze-then-unfreeze-eval", action="store_true",
                    help="增加 pretrained freeze→unfreeze 评估分支")
    ap.add_argument("--discriminative-lr-eval", action="store_true",
                    help="增加 pretrained discriminative learning rate 评估分支")
    ap.add_argument("--partial-unfreeze-eval", action="store_true",
                    help="增加 pretrained partial unfreeze 评估分支")
    args = ap.parse_args()

    if args.freeze_then_unfreeze_eval and args.freeze_backbone_epochs <= 0:
        raise SystemExit("启用 --freeze-then-unfreeze-eval 时，--freeze-backbone-epochs 必须 > 0")
    if args.partial_unfreeze_eval and args.unfreeze_last_n_layers <= 0:
        raise SystemExit("启用 --partial-unfreeze-eval 时，--unfreeze-last-n-layers 必须 > 0")
    if args.discriminative_lr_eval and math.isclose(args.backbone_lr_scale, args.head_lr_scale, rel_tol=1e-12, abs_tol=1e-12):
        raise SystemExit("启用 --discriminative-lr-eval 时，请设置不同的 --backbone-lr-scale 与 --head-lr-scale")

    ht_runtime = None
    ht_runtime_error = None
    if args.use_hypertraps:
        try:
            ht_runtime = resolve_hypertraps_runtime(
                Path(ROOT),
                args.ht_bin or None,
                args.ht_post_process or None,
            )
            msg = f"[two_stage_eval][ht] backend={ht_runtime['backend']} bin={ht_runtime['bin']}"
            if ht_runtime.get("post_process"):
                msg += f" post_process={ht_runtime['post_process']}"
            print(msg)
        except Exception as exc:
            ht_runtime_error = str(exc)
            print(f"[two_stage_eval][ht][warn] {ht_runtime_error}")

    os.makedirs(args.outdir, exist_ok=True)
    fold_root = os.path.join(args.outdir, "fold_runs")
    os.makedirs(fold_root, exist_ok=True)

    setup = prepare_feature_setup(
        transitions_path=args.transitions,
        hierarchy_path=args.hierarchy,
        no_meta=args.no_meta,
        meta_fields=args.meta_fields,
    )
    trans_all = setup.transitions

    # ---- loss 侧候选掩码: Stage 2 head 与 evaluator 共用同一份可逆先验 ----
    ts_loss_mask, loss_mask_info = resolve_loss_setup(setup, args)
    n_loss_edges = sum(1 for t in trans_all if len(t.get("lost", []) or []) > 0)
    loss_side_available = (ts_loss_mask is not None) and n_loss_edges > 0

    if args.eval_sides == "gain":
        sides = [GAIN]
        loss_mask_info["eval_side_decision"] = "disabled by --eval-sides gain"
    elif args.eval_sides == "both":
        if not loss_side_available:
            raise SystemExit(
                "--eval-sides both 需要: (1) transitions 中存在 loss 边, "
                "(2) feature_hierarchy.csv 提供可逆先验以界定 loss 候选。"
                f" 当前 n_loss_edges={n_loss_edges}, reason={loss_mask_info.get('reason')}"
            )
        sides = [GAIN, LOSS]
        loss_mask_info["eval_side_decision"] = "enabled by --eval-sides both"
    else:
        sides = [GAIN, LOSS] if loss_side_available else [GAIN]
        loss_mask_info["eval_side_decision"] = (
            "auto -> gain+loss" if loss_side_available
            else f"auto -> gain only ({loss_mask_info.get('reason') or 'no loss edges'})"
        )
    loss_mask_info["eval_sides"] = list(sides)

    # Stage 2 head 与评估通道解耦
    ts_head_mode = "gain_only"
    if args.ts_head_mode == "auto" and loss_side_available:
        ts_head_mode = "dual_head_selective_reversible"
    if ts_head_mode == "gain_only":
        ts_loss_mask_for_model = None
    else:
        ts_loss_mask_for_model = ts_loss_mask
    loss_mask_info["ts_head_mode"] = ts_head_mode
    loss_mask_info["ts_head_mode_arg"] = args.ts_head_mode

    eval_loss_mask = ts_loss_mask if LOSS in sides else None

    print(f">>> eval sides = {sides} ({loss_mask_info['eval_side_decision']})")
    print(f">>> stage2 head_mode = {ts_head_mode} (arg={args.ts_head_mode})")
    if LOSS in sides:
        print(
            f">>> loss 候选先验: source={loss_mask_info['source']} "
            f"n_loss_eligible_features={loss_mask_info['n_loss_eligible_features']}"
        )

    # 评估边集合: 两侧共用同一套 fold 划分。sides==[gain] 时恰等于历史 gained>0 集合
    trans_eval = filter_event_transitions(trans_all, include_loss=(LOSS in sides))
    if not trans_eval:
        raise SystemExit(
            f"当前 transitions 文件中没有可用于评估的边 (sides={sides})")

    profile = transition_profile(trans_all, trans_eval, sides=sides)
    stratify_mode, stratify_info = resolve_stratify_mode(trans_eval, args.stratify_by)
    print(f">>> stratify_by = {stratify_mode} ({stratify_info['reason']}) "
          f"labels={stratify_info['labels']}")
    groups = [t["group"] for t in trans_eval]
    if args.split_mode == "grouped":
        folds = group_kfold_indices(groups, n_splits=args.folds, seed=args.seed)
    else:
        folds = random_kfold_indices(len(trans_eval), n_splits=args.folds, seed=args.seed)
    if not folds:
        raise SystemExit("无法构造有效的 fold 划分")

    method_order = ["B0_uniform", "B1_prevalence"]
    if not args.skip_random_init_control:
        method_order.append("TS_random_init")
    if args.frozen_backbone_eval and not args.skip_random_init_control:
        method_order.append("TS_random_init_frozen_backbone")
    method_order.append("TS_pretrained")
    ht_label = f"HT_{args.ht_model}" if args.use_hypertraps else None
    if ht_label:
        method_order.append(ht_label)
    if args.frozen_backbone_eval:
        method_order.append("TS_pretrained_frozen_backbone")
    if args.freeze_then_unfreeze_eval:
        method_order.append("TS_pretrained_freeze_then_unfreeze")
    if args.discriminative_lr_eval:
        method_order.append("TS_pretrained_discriminative_lr")
    if args.partial_unfreeze_eval:
        method_order.append("TS_pretrained_partial_unfreeze")
    if args.aligned_epochs > 0:
        method_order.append("TS_pretrained_aligned")

    # loss 侧参与评估的方法 (HyperTraPS 为单调累积模型, 不支持 1->0)
    loss_method_order = [
        m for m in method_order if not m.startswith(LOSS_UNSUPPORTED_PREFIXES)
    ]
    # 综合通道: 统一事件空间上的单一性能, 所有方法 (含 HT) 均参与
    combined_method_order = list(method_order)

    folds_payload: List[dict] = []
    per_item = {m: [] for m in method_order}
    per_item_loss = {m: [] for m in loss_method_order}
    per_item_combined = {m: [] for m in combined_method_order}
    # 方法级耗时埋点: 每折记录 train(建模/fit) 与 eval(测试) 的 wall-clock
    method_train_runtime: dict = {}
    method_eval_runtime: dict = {}
    pretrained_reports = []
    pretrained_frozen_reports = []
    pretrained_freeze_then_unfreeze_reports = []
    pretrained_discriminative_lr_reports = []
    pretrained_partial_unfreeze_reports = []
    aligned_reports = []
    random_reports = []
    random_frozen_reports = []

    print(
        f">>> input={len(trans_all)} eval={len(trans_eval)} features={setup.L} "
        f"split_mode={args.split_mode} groups={len(set(groups))} folds={len(folds)} "
        f"gain_edges={profile['n_gain_side_edges']} loss_edges={profile['n_loss_side_edges']}"
    )

    for fi, (tr_idx, te_idx) in enumerate(folds):
        train = [trans_eval[i] for i in tr_idx]
        test = [trans_eval[i] for i in te_idx]
        fold_dir = os.path.join(fold_root, f"fold_{fi:02d}")
        os.makedirs(fold_dir, exist_ok=True)
        print(f"\n=== Fold {fi + 1}/{len(folds)}: train={len(train)} test={len(test)} ===")

        b0 = UniformBaseline(setup.L).fit(train)
        b1 = PrevalenceBaseline(setup.L).fit(train)
        predictors = {
            "B0_uniform": b0.predict_fn,
            "B1_prevalence": b1.predict_fn,
        }
        # loss 侧预测器: B0 与 gain 侧共用 (全 1, 由 evaluator 在 loss 候选上归一);
        # B1 换成"被丢失特征"频次, 与 gain 侧同构的边际频率对照。
        predictors_loss: dict = {}
        if LOSS in sides:
            b1l = LossPrevalenceBaseline(setup.L).fit(train)
            predictors_loss["B0_uniform"] = b0.predict_fn
            predictors_loss["B1_prevalence"] = b1l.predict_fn

        def _reg(label: str, model) -> None:
            """注册某个 TS 变体的两侧预测器。"""
            predictors[label] = model.predict_fn
            if LOSS in sides and model.fit_report.get("loss_head_enabled"):
                predictors_loss[label] = model.predict_loss_fn

        if ht_label:
            if ht_runtime is None:
                print(f"  [ht] WARN: {ht_runtime_error}; skip HT on this fold")
            else:
                try:
                    # HyperTraPS 只支持单调累积: 只喂含 gain 的边, 否则 loss-only
                    # 边会退化成 src==dst 的自环样本, 污染 MCMC
                    _t0 = time.perf_counter()
                    predictors[ht_label] = build_and_run_hypertraps_predictor(
                        filter_gain_transitions(train),
                        setup.L,
                        runtime=ht_runtime,
                        model=args.ht_model,
                        N=args.ht_N,
                        burn=args.ht_burn,
                        r=args.ht_r,
                        timeout_sec=args.ht_timeout,
                        threads=args.ht_threads,
                    )
                    method_train_runtime.setdefault(ht_label, []).append(
                        time.perf_counter() - _t0)
                except Exception as exc:
                    print(f"  [ht] WARN: HyperTraPS fold {fi} failed: {exc}; skip HT on this fold")

        if not args.skip_random_init_control:
            random_model = _build_model(args, setup, args.seed + fi,
                                        head_mode=ts_head_mode, loss_mask=ts_loss_mask_for_model)
            random_init_state = copy.deepcopy(random_model.net.state_dict())
            _t0 = time.perf_counter()
            random_fit = random_model.fit_transitions(train, epochs=args.finetune_epochs, pretrained=False)
            method_train_runtime.setdefault("TS_random_init", []).append(
                time.perf_counter() - _t0)
            _reg("TS_random_init", random_model)
            rr = dict(random_fit)
            rr.update({"fold": fi, "setup": "random_init"})
            random_reports.append(rr)
            with open(os.path.join(fold_dir, "random_init_stage2_report.json"), "w") as f:
                json.dump(_jsonable(rr), f, indent=2, ensure_ascii=False)

            if args.frozen_backbone_eval:
                random_frozen_model = _build_model(args, setup, args.seed + fi,
                                        head_mode=ts_head_mode, loss_mask=ts_loss_mask_for_model)
                random_frozen_model.net.load_state_dict(random_init_state, strict=False)
                random_frozen_fit = random_frozen_model.fit_transitions(
                    train,
                    epochs=args.finetune_epochs,
                    pretrained=False,
                    freeze_backbone=True,
                )
                _reg("TS_random_init_frozen_backbone", random_frozen_model)
                rfr = dict(random_frozen_fit)
                rfr.update({"fold": fi, "setup": "random_init_frozen_backbone"})
                random_frozen_reports.append(rfr)
                with open(os.path.join(fold_dir, "random_init_frozen_backbone_stage2_report.json"), "w") as f:
                    json.dump(_jsonable(rfr), f, indent=2, ensure_ascii=False)

        train_states = build_state_examples(
            train,
            state_source=args.state_source,
            dedupe=(not args.no_dedupe_states),
        )

        pretrained_model = _build_model(args, setup, args.seed + fi,
                                        head_mode=ts_head_mode, loss_mask=ts_loss_mask_for_model)
        # Stage 1 是 EvoState 的真实成本的一部分, 单独计时后再和 stage 2 相加,
        # 否则 "EvoState vs HyperTraPS" 的加速比会漏掉预训练那一段。
        _t0 = time.perf_counter()
        stage1_report = pretrained_model.pretrain(
            train_states,
            epochs=args.pretrain_epochs,
            mask_prob=args.mask_prob,
            mask_prob_pos=args.mask_prob_pos,
            mask_prob_neg=args.mask_prob_neg,
            contrastive_weight=args.contrastive_weight,
            contrastive_temperature=args.contrastive_temperature,
            contrastive_warmup_epochs=args.contrastive_warmup_epochs,
            similar_state_neg_threshold=args.similar_state_neg_threshold,
            checkpoint_metric=args.pretrain_checkpoint_metric,
        )
        method_train_runtime.setdefault("TS_pretrained_stage1", []).append(
            time.perf_counter() - _t0)
        pretrained_stage1_state = copy.deepcopy(pretrained_model.net.state_dict())
        _t0 = time.perf_counter()
        stage2_report = pretrained_model.fit_transitions(
            train,
            epochs=args.finetune_epochs,
            pretrained=True,
        )
        method_train_runtime.setdefault("TS_pretrained", []).append(
            time.perf_counter() - _t0)
        _reg("TS_pretrained", pretrained_model)
        pr = {
            "fold": fi,
            "n_states": int(len(train_states)),
            "state_source": args.state_source,
            "dedupe_states": bool(not args.no_dedupe_states),
            "pretrain_report": stage1_report,
            "stage2_fit_report": stage2_report,
        }
        pretrained_reports.append(pr)
        with open(os.path.join(fold_dir, "pretrained_report.json"), "w") as f:
            json.dump(_jsonable(pr), f, indent=2, ensure_ascii=False)

        if args.dump_fold_models:
            # 折外条件预测分析需要对"不加背景 / 错配背景"这类合成状态打分, 折外
            # 预测 npz 只覆盖真实转移的 src, 因此额外落盘该折的 stage2 权重,
            # 并记录该折留出的 group_id, 供分析脚本按留出关系取用对应折的模型。
            pretrained_model.save(os.path.join(fold_dir, "stage2_model.pt"),
                                  extra={"fold": fi})
            with open(os.path.join(fold_dir, "holdout_groups.json"), "w") as f:
                json.dump({"fold": fi,
                           "holdout_groups": sorted({str(t["group"]) for t in test}),
                           "train_groups": sorted({str(t["group"]) for t in train})},
                          f, indent=2, ensure_ascii=False)

        if args.frozen_backbone_eval:
            frozen_model = _build_model(args, setup, args.seed + fi,
                                        head_mode=ts_head_mode, loss_mask=ts_loss_mask_for_model)
            frozen_model.net.load_state_dict(pretrained_stage1_state, strict=False)
            frozen_model.pretrain_report = copy.deepcopy(stage1_report)
            _t0 = time.perf_counter()
            frozen_stage2_report = frozen_model.fit_transitions(
                train,
                epochs=args.finetune_epochs,
                pretrained=True,
                freeze_backbone=True,
            )
            method_train_runtime.setdefault("TS_pretrained_frozen_backbone", []).append(
                time.perf_counter() - _t0)
            _reg("TS_pretrained_frozen_backbone", frozen_model)
            pfr = {
                "fold": fi,
                "n_states": int(len(train_states)),
                "state_source": args.state_source,
                "dedupe_states": bool(not args.no_dedupe_states),
                "pretrain_report": copy.deepcopy(stage1_report),
                "stage2_fit_report": frozen_stage2_report,
            }
            pretrained_frozen_reports.append(pfr)
            with open(os.path.join(fold_dir, "pretrained_frozen_backbone_report.json"), "w") as f:
                json.dump(_jsonable(pfr), f, indent=2, ensure_ascii=False)

        if args.freeze_then_unfreeze_eval:
            freeze_then_unfreeze_model = _build_model(args, setup, args.seed + fi,
                                        head_mode=ts_head_mode, loss_mask=ts_loss_mask_for_model)
            freeze_then_unfreeze_model.net.load_state_dict(pretrained_stage1_state, strict=False)
            freeze_then_unfreeze_model.pretrain_report = copy.deepcopy(stage1_report)
            _t0 = time.perf_counter()
            freeze_then_unfreeze_stage2_report = freeze_then_unfreeze_model.fit_transitions(
                train,
                epochs=args.finetune_epochs,
                pretrained=True,
                freeze_backbone_epochs=args.freeze_backbone_epochs,
                backbone_lr_scale=args.backbone_lr_scale,
                head_lr_scale=args.head_lr_scale,
            )
            method_train_runtime.setdefault("TS_pretrained_freeze_then_unfreeze", []).append(
                time.perf_counter() - _t0)
            _reg("TS_pretrained_freeze_then_unfreeze", freeze_then_unfreeze_model)
            fur = {
                "fold": fi,
                "n_states": int(len(train_states)),
                "state_source": args.state_source,
                "dedupe_states": bool(not args.no_dedupe_states),
                "pretrain_report": copy.deepcopy(stage1_report),
                "stage2_fit_report": freeze_then_unfreeze_stage2_report,
            }
            pretrained_freeze_then_unfreeze_reports.append(fur)
            with open(os.path.join(fold_dir, "pretrained_freeze_then_unfreeze_report.json"), "w") as f:
                json.dump(_jsonable(fur), f, indent=2, ensure_ascii=False)

        if args.discriminative_lr_eval:
            discriminative_lr_model = _build_model(args, setup, args.seed + fi,
                                        head_mode=ts_head_mode, loss_mask=ts_loss_mask_for_model)
            discriminative_lr_model.net.load_state_dict(pretrained_stage1_state, strict=False)
            discriminative_lr_model.pretrain_report = copy.deepcopy(stage1_report)
            _t0 = time.perf_counter()
            discriminative_lr_stage2_report = discriminative_lr_model.fit_transitions(
                train,
                epochs=args.finetune_epochs,
                pretrained=True,
                backbone_lr_scale=args.backbone_lr_scale,
                head_lr_scale=args.head_lr_scale,
            )
            method_train_runtime.setdefault("TS_pretrained_discriminative_lr", []).append(
                time.perf_counter() - _t0)
            _reg("TS_pretrained_discriminative_lr", discriminative_lr_model)
            dlr = {
                "fold": fi,
                "n_states": int(len(train_states)),
                "state_source": args.state_source,
                "dedupe_states": bool(not args.no_dedupe_states),
                "pretrain_report": copy.deepcopy(stage1_report),
                "stage2_fit_report": discriminative_lr_stage2_report,
            }
            pretrained_discriminative_lr_reports.append(dlr)
            with open(os.path.join(fold_dir, "pretrained_discriminative_lr_report.json"), "w") as f:
                json.dump(_jsonable(dlr), f, indent=2, ensure_ascii=False)

        if args.partial_unfreeze_eval:
            partial_unfreeze_model = _build_model(args, setup, args.seed + fi,
                                        head_mode=ts_head_mode, loss_mask=ts_loss_mask_for_model)
            partial_unfreeze_model.net.load_state_dict(pretrained_stage1_state, strict=False)
            partial_unfreeze_model.pretrain_report = copy.deepcopy(stage1_report)
            _t0 = time.perf_counter()
            partial_unfreeze_stage2_report = partial_unfreeze_model.fit_transitions(
                train,
                epochs=args.finetune_epochs,
                pretrained=True,
                unfreeze_last_n_layers=args.unfreeze_last_n_layers,
                backbone_lr_scale=args.backbone_lr_scale,
                head_lr_scale=args.head_lr_scale,
            )
            method_train_runtime.setdefault("TS_pretrained_partial_unfreeze", []).append(
                time.perf_counter() - _t0)
            _reg("TS_pretrained_partial_unfreeze", partial_unfreeze_model)
            pur = {
                "fold": fi,
                "n_states": int(len(train_states)),
                "state_source": args.state_source,
                "dedupe_states": bool(not args.no_dedupe_states),
                "pretrain_report": copy.deepcopy(stage1_report),
                "stage2_fit_report": partial_unfreeze_stage2_report,
            }
            pretrained_partial_unfreeze_reports.append(pur)
            with open(os.path.join(fold_dir, "pretrained_partial_unfreeze_report.json"), "w") as f:
                json.dump(_jsonable(pur), f, indent=2, ensure_ascii=False)

        if args.aligned_epochs > 0:
            aligned_model = _build_model(args, setup, args.seed + fi,
                                        head_mode=ts_head_mode, loss_mask=ts_loss_mask_for_model)
            aligned_stage1_report = aligned_model.pretrain(
                train_states,
                epochs=args.pretrain_epochs,
                mask_prob=args.mask_prob,
                mask_prob_pos=args.mask_prob_pos,
                mask_prob_neg=args.mask_prob_neg,
                contrastive_weight=args.contrastive_weight,
                contrastive_temperature=args.contrastive_temperature,
                contrastive_warmup_epochs=args.contrastive_warmup_epochs,
                similar_state_neg_threshold=args.similar_state_neg_threshold,
                checkpoint_metric=args.pretrain_checkpoint_metric,
            )
            aligned_stage15_report = aligned_model.align_transitions(
                train,
                epochs=args.aligned_epochs,
                lr_scale=args.aligned_lr_scale,
            )
            _t0 = time.perf_counter()
            aligned_stage2_report = aligned_model.fit_transitions(
                train,
                epochs=args.finetune_epochs,
                pretrained=True,
            )
            method_train_runtime.setdefault("TS_pretrained_aligned", []).append(
                time.perf_counter() - _t0)
            _reg("TS_pretrained_aligned", aligned_model)
            ar = {
                "fold": fi,
                "n_states": int(len(train_states)),
                "state_source": args.state_source,
                "dedupe_states": bool(not args.no_dedupe_states),
                "pretrain_report": aligned_stage1_report,
                "alignment_report": aligned_stage15_report,
                "stage2_fit_report": aligned_stage2_report,
            }
            aligned_reports.append(ar)
            with open(os.path.join(fold_dir, "pretrained_aligned_report.json"), "w") as f:
                json.dump(_jsonable(ar), f, indent=2, ensure_ascii=False)

        fold_row = {
            "fold": fi,
            "n_train": int(len(train)),
            "n_test": int(len(test)),
            "n_train_gain": int(sum(1 for t in train if len(t.get("gained", []) or []) > 0)),
            "n_train_loss": int(sum(1 for t in train if len(t.get("lost", []) or []) > 0)),
            "n_test_gain": int(sum(1 for t in test if len(t.get("gained", []) or []) > 0)),
            "n_test_loss": int(sum(1 for t in test if len(t.get("lost", []) or []) > 0)),
            "methods": {},
        }
        for method in method_order:
            if method not in predictors:
                continue
            _t0 = time.perf_counter()
            ev = eval_transitions(test, predictors[method], side=GAIN)
            method_eval_runtime.setdefault(method, []).append(
                time.perf_counter() - _t0)
            sm = summarize(ev)
            fold_row["methods"][method] = _jsonable(sm)
            per_item[method].append(np.asarray(ev["per_item_logl"], dtype=float))
        attach_logl_lift(fold_row["methods"])
        if args.dump_predictions:
            _pred_dir = os.path.join(args.outdir, "predictions")
            os.makedirs(_pred_dir, exist_ok=True)
            _pp = dump_fold_predictions(
                os.path.join(_pred_dir, f"predictions_fold_{fi:02d}.npz"),
                test, predictors, setup.L)
            if _pp:
                print(f"  [dump] out-of-fold predictions -> {_pp}")
        for method in method_order:
            if method in fold_row["methods"]:
                print("  " + fmt_metrics(method, fold_row["methods"][method], side=GAIN))

        if LOSS in sides:
            fold_row["methods_loss"] = {}
            for method in loss_method_order:
                if method not in predictors_loss:
                    continue
                ev_l = eval_transitions(test, predictors_loss[method],
                                        side=LOSS, loss_mask=eval_loss_mask)
                sm_l = summarize(ev_l)
                fold_row["methods_loss"][method] = _jsonable(sm_l)
                per_item_loss[method].append(np.asarray(ev_l["per_item_logl"], dtype=float))
            attach_logl_lift(fold_row["methods_loss"])
            for method in loss_method_order:
                if method in fold_row["methods_loss"]:
                    print("  " + fmt_metrics(method, fold_row["methods_loss"][method], side=LOSS))

            # ---- 综合通道: 统一事件空间, 单一综合性能 ----
            pi = estimate_gain_event_share(train)  # 只用训练折, 防泄漏
            fold_row["pi_gain"] = float(pi)
            fold_row["methods_combined"] = {}
            for method in combined_method_order:
                if method not in predictors:
                    continue
                # 无 loss 预测能力的方法 (HT / 单头 TS) 用该侧 uniform 兜底
                lfn = predictors_loss.get(method, b0.predict_fn)
                ev_c = eval_transitions_combined(
                    test, predictors[method], loss_predict_fn=lfn,
                    loss_mask=eval_loss_mask, pi=pi)
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
                sides, eval_loss_mask, b0.predict_fn,
                pooled_pi=fold_row.get("pi_gain"))

        folds_payload.append(fold_row)

    metrics_nested, overall_by_method = aggregate_fold_metrics(folds_payload, method_order)
    metrics_loss_nested: dict = {}
    overall_loss_by_method: dict = {}
    metrics_combined_nested: dict = {}
    overall_combined_by_method: dict = {}
    if LOSS in sides:
        metrics_loss_nested, overall_loss_by_method = aggregate_fold_metrics(
            folds_payload, loss_method_order, methods_key="methods_loss")
        metrics_combined_nested, overall_combined_by_method = aggregate_fold_metrics(
            folds_payload, combined_method_order, methods_key="methods_combined")
    comparisons = {
        "pretrained_vs_prevalence": _compare_if_available(per_item, "TS_pretrained", "B1_prevalence", "TS_pretrained", "Prevalence", args.seed),
        "pretrained_vs_random": None if args.skip_random_init_control else _compare_if_available(per_item, "TS_pretrained", "TS_random_init", "TS_pretrained", "TS_random_init", args.seed + 1),
        "pretrained_frozen_vs_pretrained": _compare_if_available(per_item, "TS_pretrained_frozen_backbone", "TS_pretrained", "TS_pretrained_frozen_backbone", "TS_pretrained", args.seed + 2),
        "pretrained_frozen_vs_random_frozen": None if args.skip_random_init_control else _compare_if_available(per_item, "TS_pretrained_frozen_backbone", "TS_random_init_frozen_backbone", "TS_pretrained_frozen_backbone", "TS_random_init_frozen_backbone", args.seed + 3),
        "freeze_then_unfreeze_vs_pretrained": _compare_if_available(per_item, "TS_pretrained_freeze_then_unfreeze", "TS_pretrained", "TS_pretrained_freeze_then_unfreeze", "TS_pretrained", args.seed + 4),
        "freeze_then_unfreeze_vs_frozen": _compare_if_available(per_item, "TS_pretrained_freeze_then_unfreeze", "TS_pretrained_frozen_backbone", "TS_pretrained_freeze_then_unfreeze", "TS_pretrained_frozen_backbone", args.seed + 5),
        "discriminative_lr_vs_pretrained": _compare_if_available(per_item, "TS_pretrained_discriminative_lr", "TS_pretrained", "TS_pretrained_discriminative_lr", "TS_pretrained", args.seed + 6),
        "discriminative_lr_vs_frozen": _compare_if_available(per_item, "TS_pretrained_discriminative_lr", "TS_pretrained_frozen_backbone", "TS_pretrained_discriminative_lr", "TS_pretrained_frozen_backbone", args.seed + 7),
        "partial_unfreeze_vs_pretrained": _compare_if_available(per_item, "TS_pretrained_partial_unfreeze", "TS_pretrained", "TS_pretrained_partial_unfreeze", "TS_pretrained", args.seed + 8),
        "partial_unfreeze_vs_frozen": _compare_if_available(per_item, "TS_pretrained_partial_unfreeze", "TS_pretrained_frozen_backbone", "TS_pretrained_partial_unfreeze", "TS_pretrained_frozen_backbone", args.seed + 9),
        "aligned_vs_pretrained": None if args.aligned_epochs <= 0 else _compare_if_available(per_item, "TS_pretrained_aligned", "TS_pretrained", "TS_pretrained_aligned", "TS_pretrained", args.seed + 10),
        "aligned_vs_random": None if (args.aligned_epochs <= 0 or args.skip_random_init_control) else _compare_if_available(per_item, "TS_pretrained_aligned", "TS_random_init", "TS_pretrained_aligned", "TS_random_init", args.seed + 11),
        "ht_vs_prevalence": None if not ht_label else _compare_if_available(per_item, ht_label, "B1_prevalence", "HyperTraPS", "Prevalence", args.seed + 12),
        "pretrained_vs_hypertraps": None if not ht_label else _compare_if_available(per_item, "TS_pretrained", ht_label, "TS_pretrained", "HyperTraPS", args.seed + 13),
    }
    loss_comparisons = {}
    combined_comparisons = {}
    if LOSS in sides:
        loss_comparisons = {
            "pretrained_vs_prevalence_loss": _compare_if_available(
                per_item_loss, "TS_pretrained", "B1_prevalence",
                "TS_pretrained", "LossPrevalence", args.seed + 20),
            "pretrained_vs_uniform_loss": _compare_if_available(
                per_item_loss, "TS_pretrained", "B0_uniform",
                "TS_pretrained", "Uniform", args.seed + 21),
            "pretrained_vs_random_loss": None if args.skip_random_init_control else _compare_if_available(
                per_item_loss, "TS_pretrained", "TS_random_init",
                "TS_pretrained", "TS_random_init", args.seed + 22),
        }
        combined_comparisons = {
            "pretrained_vs_prevalence_combined": _compare_if_available(
                per_item_combined, "TS_pretrained", "B1_prevalence",
                "TS_pretrained", "Prevalence", args.seed + 30),
            "pretrained_vs_uniform_combined": _compare_if_available(
                per_item_combined, "TS_pretrained", "B0_uniform",
                "TS_pretrained", "Uniform", args.seed + 31),
            "pretrained_vs_hypertraps_combined": None if not ht_label else _compare_if_available(
                per_item_combined, "TS_pretrained", ht_label,
                "TS_pretrained", "HyperTraPS", args.seed + 32),
        }

    with open(os.path.join(args.outdir, "fold_metrics.json"), "w") as f:
        json.dump(_jsonable({
            "transitions": os.path.abspath(args.transitions),
            "split_mode": args.split_mode,
            "eval_sides": list(sides),
            "transition_profile": profile,
            "folds": folds_payload,
        }), f, indent=2, ensure_ascii=False)

    summary = {
        "transitions": os.path.abspath(args.transitions),
        "transitions_path": os.path.abspath(args.transitions),
        "transition_profile": profile,
        "transition_input": setup.transition_info,
        "split_mode": args.split_mode,
        "n_transitions": int(len(trans_all)),
        "n_eval_transitions": int(len(trans_eval)),
        # 历史字段: gain-only 口径下被跳过的边数 (与 eval_sides 无关, 便于跨版本对比)
        "n_skipped_non_gain_transitions": int(profile["n_skipped_non_gain"]),
        "ht_model": (args.ht_model if args.use_hypertraps else None),
        "ht_runtime": ({
            "enabled": bool(args.use_hypertraps),
            "requested_bin": (os.path.abspath(args.ht_bin) if args.ht_bin else None),
            "requested_post_process": (os.path.abspath(args.ht_post_process) if args.ht_post_process else None),
            "backend": (None if ht_runtime is None else ht_runtime["backend"]),
            "bin": (None if ht_runtime is None else ht_runtime["bin"]),
            "post_process": (None if ht_runtime is None else ht_runtime.get("post_process")),
            "resolution_error": ht_runtime_error,
        } if args.use_hypertraps else None),
        "L": int(setup.L),
        "n_features": int(setup.L),
        "n_groups": int(len(set(groups))),
        "n_groups_input": int(len({t['group'] for t in trans_all})),
        "folds": int(len(folds)),
        "n_folds": int(len(folds)),
        # 顶层 methods/metrics/overall 始终为 gain 侧, 与历史 gain-only 口径一致
        "methods": [m for m in method_order if m in overall_by_method],
        "metrics": metrics_nested,
        "overall": overall_by_method,
        # 方案B 双通道: loss 侧单列, 只与其自身基线对照, 不与 gain 侧并列比较
        "eval_sides": list(sides),
        "eval_sides_arg": args.eval_sides,
        "loss_side_eval": {
            "enabled": bool(LOSS in sides),
            "candidate_mask": loss_mask_info,
            "stage2_head_mode": ts_head_mode,
            "loss_weight": float(args.loss_weight),
            "methods": [m for m in loss_method_order if m in overall_loss_by_method],
            "excluded_methods": [
                m for m in method_order if m.startswith(LOSS_UNSUPPORTED_PREFIXES)
            ],
            "excluded_reason": "HyperTraPS 为单调累积模型, 不支持 1->0 事件",
            "metrics": metrics_loss_nested,
            "overall": overall_loss_by_method,
            "semantics": (
                "loss 侧的 1->0 为端点净状态消失/回退, 不等于已确证的回复突变事件; "
                "候选集合限定为 src==1 且属 hierarchy 可逆先验的位。"
                "两侧候选空间规模差异大, 指标不作跨侧并列比较。"
            ),
            **loss_comparisons,
        },
        # 综合通道: 统一事件空间上的单一性能, 双头模型的 headline 指标
        "combined_eval": {
            "enabled": bool(LOSS in sides),
            "pi_gain_per_fold": [f.get("pi_gain") for f in folds_payload],
            "methods": [m for m in combined_method_order if m in overall_combined_by_method],
            "metrics": metrics_combined_nested,
            "overall": overall_combined_by_method,
            "definition": (
                "事件空间 = {(gain,i): s_i==0} ∪ {(loss,j): s_j==1 且 j 属可逆先验}; "
                "P(gain,i)=pi*P_g(i), P(loss,j)=(1-pi)*P_l(j)。pi 仅由训练折的 "
                "gain/loss 事件比估计。无 loss 预测能力的方法 (HyperTraPS / 单头 TS) "
                "在 loss 侧以该侧 uniform 兜底。loss 候选为空或 pi==1 时本指标恒等于 "
                "gain 侧指标, 故 gain-only 数据集上与历史结果同口径。"
            ),
            **combined_comparisons,
        },
        # 分层评估: 池化训练 + 分层报告 (分层变量不进模型)
        "stratified_eval": {
            "enabled": bool(stratify_mode != "none"),
            "mode": stratify_mode,
            "info": stratify_info,
            "note": (
                "fold 划分与训练均为池化, 分层只发生在评估阶段; 分层变量 (如抗生素) "
                "不作为模型输入。combined 通道的 pi 优先按该层训练子集估计。"
            ),
            "strata": (
                aggregate_strata(
                    folds_payload, stratify_info["labels"],
                    {"gain": method_order, "loss": loss_method_order,
                     "combined": combined_method_order})
                if stratify_mode != "none" else {}
            ),
        },
        "two_stage_setup": {
            "arch": args.arch,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "dropout": args.dropout,
            "weight_decay": args.weight_decay,
            "lr": args.lr,
            "batch_size": args.batch_size,
            "val_frac": args.val_frac,
            "patience": args.patience,
            "pretrain_epochs": args.pretrain_epochs,
            "finetune_epochs": args.finetune_epochs,
            "aligned_epochs": args.aligned_epochs,
            "aligned_lr_scale": args.aligned_lr_scale,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "backbone_lr_scale": args.backbone_lr_scale,
            "head_lr_scale": args.head_lr_scale,
            "unfreeze_last_n_layers": args.unfreeze_last_n_layers,
            "freeze_then_unfreeze_eval": bool(args.freeze_then_unfreeze_eval),
            "discriminative_lr_eval": bool(args.discriminative_lr_eval),
            "partial_unfreeze_eval": bool(args.partial_unfreeze_eval),
            "mask_prob": args.mask_prob,
            "mask_prob_pos": args.mask_prob_pos,
            "mask_prob_neg": args.mask_prob_neg,
            "contrastive_weight": args.contrastive_weight,
            "contrastive_temperature": args.contrastive_temperature,
            "contrastive_warmup_epochs": args.contrastive_warmup_epochs,
            "similar_state_neg_threshold": args.similar_state_neg_threshold,
            "pretrain_checkpoint_metric": args.pretrain_checkpoint_metric,
            "state_source": args.state_source,
            "dedupe_states": bool(not args.no_dedupe_states),
            "frozen_backbone_eval": bool(args.frozen_backbone_eval),
            "feat_meta": setup.meta_info,
        },
        "pretrained_reports": pretrained_reports,
        "pretrained_frozen_backbone_reports": pretrained_frozen_reports,
        "pretrained_freeze_then_unfreeze_reports": pretrained_freeze_then_unfreeze_reports,
        "pretrained_discriminative_lr_reports": pretrained_discriminative_lr_reports,
        "pretrained_partial_unfreeze_reports": pretrained_partial_unfreeze_reports,
        "aligned_reports": aligned_reports,
        "random_init_reports": random_reports,
        "random_init_frozen_backbone_reports": random_frozen_reports,
        "pretrained_mean_train_val_gap": _nanmean([r["stage2_fit_report"].get("train_val_gap", float("nan")) for r in pretrained_reports]),
        "pretrained_frozen_mean_train_val_gap": _nanmean([r["stage2_fit_report"].get("train_val_gap", float("nan")) for r in pretrained_frozen_reports]),
        "pretrained_freeze_then_unfreeze_mean_train_val_gap": _nanmean([r["stage2_fit_report"].get("train_val_gap", float("nan")) for r in pretrained_freeze_then_unfreeze_reports]),
        "pretrained_discriminative_lr_mean_train_val_gap": _nanmean([r["stage2_fit_report"].get("train_val_gap", float("nan")) for r in pretrained_discriminative_lr_reports]),
        "pretrained_partial_unfreeze_mean_train_val_gap": _nanmean([r["stage2_fit_report"].get("train_val_gap", float("nan")) for r in pretrained_partial_unfreeze_reports]),
        "aligned_mean_train_val_gap": _nanmean([r["stage2_fit_report"].get("train_val_gap", float("nan")) for r in aligned_reports]),
        **comparisons,
    }

    # ---- 方法级耗时聚合 + 导出 method_runtime.csv ----
    _all_rt_methods = []
    for _d in (method_train_runtime, method_eval_runtime):
        for _m in _d:
            if _m not in _all_rt_methods:
                _all_rt_methods.append(_m)
    method_runtime_summary = {}
    for _m in _all_rt_methods:
        _tr = method_train_runtime.get(_m, [])
        _ev = method_eval_runtime.get(_m, [])
        _n = max(len(_tr), len(_ev))
        _tot = [
            (_tr[i] if i < len(_tr) else 0.0) + (_ev[i] if i < len(_ev) else 0.0)
            for i in range(_n)
        ]
        method_runtime_summary[_m] = {
            "n_folds": int(_n),
            "mean_train_sec": (float(np.mean(_tr)) if _tr else None),
            "mean_eval_sec": (float(np.mean(_ev)) if _ev else None),
            "mean_total_sec": (float(np.mean(_tot)) if _tot else None),
            "train_sec_per_fold": [float(x) for x in _tr],
            "eval_sec_per_fold": [float(x) for x in _ev],
        }
    summary["method_runtime_sec"] = method_runtime_summary
    _rt_csv = os.path.join(args.outdir, "method_runtime.csv")
    with open(_rt_csv, "w", newline="") as _f:
        _w = csv.writer(_f)
        _w.writerow(["method", "n_folds", "mean_train_sec", "mean_eval_sec", "mean_total_sec"])
        for _m in _all_rt_methods:
            _s = method_runtime_summary[_m]
            _w.writerow([
                _m, _s["n_folds"],
                ("" if _s["mean_train_sec"] is None else f"{_s['mean_train_sec']:.4f}"),
                ("" if _s["mean_eval_sec"] is None else f"{_s['mean_eval_sec']:.4f}"),
                ("" if _s["mean_total_sec"] is None else f"{_s['mean_total_sec']:.4f}"),
            ])
    print(f"[two_stage_eval] method_runtime.csv: {_rt_csv}")
    print("\n=== Method runtime (sec) ===")
    for _m in _all_rt_methods:
        _s = method_runtime_summary[_m]
        print(f"  {_m:40s} train={_s['mean_train_sec']} eval={_s['mean_eval_sec']} "
              f"total={_s['mean_total_sec']}")

    with open(os.path.join(args.outdir, "eval_summary.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=2, ensure_ascii=False)

    print("\n=== Overall (gain side) ===")
    for method in summary["methods"]:
        print("  " + fmt_metrics(method, overall_by_method[method], side=GAIN))
    if LOSS in sides:
        print("\n=== Overall (loss side) ===")
        for method in loss_method_order:
            if method in overall_loss_by_method:
                print("  " + fmt_metrics(method, overall_loss_by_method[method], side=LOSS))
        for key in ("pretrained_vs_uniform_loss", "pretrained_vs_prevalence_loss"):
            if loss_comparisons.get(key):
                print(f"\n=== [loss] {key} ===")
                print(loss_comparisons[key])
        print("\n=== Overall (combined) ===")
        for method in combined_method_order:
            if method in overall_combined_by_method:
                print("  " + fmt_metrics(
                    method, overall_combined_by_method[method], side=COMBINED))
        for key in ("pretrained_vs_uniform_combined", "pretrained_vs_prevalence_combined"):
            if combined_comparisons.get(key):
                print(f"\n=== [combined] {key} ===")
                print(combined_comparisons[key])
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

    print(f"[two_stage_eval] summary: {os.path.join(args.outdir, 'eval_summary.json')}")


if __name__ == "__main__":
    main()
