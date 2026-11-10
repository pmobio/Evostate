#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import os


def _load_two_stage_backbone_module():
    here = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(here, "two_stage_backbone.py")
    module_name = "two_stage_backbone_runtime"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载 two_stage_backbone 模块: {module_path}")
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TSB = _load_two_stage_backbone_module()
TwoStageTransitionModel = _TSB.TwoStageTransitionModel
_jsonable = _TSB._jsonable
filter_gain_transitions = _TSB.filter_gain_transitions
first_step_ranking_table = _TSB.first_step_ranking_table
prepare_feature_setup = _TSB.prepare_feature_setup


def main():
    ap = argparse.ArgumentParser(description="Stage 2 next-event transition fine-tuning")
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--pretrained", default="")
    ap.add_argument("--arch", default="transformer", choices=["mlp", "transformer"])
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hierarchy", default=None)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--meta-fields", default="")
    ap.add_argument("--freeze-backbone", action="store_true",
                    help="冻结 backbone，仅训练 stage2 gain 线性头")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=0,
                    help="先冻结 backbone 训练若干 epoch，再自动解冻继续训练")
    ap.add_argument("--backbone-lr-scale", type=float, default=1.0,
                    help="Stage 2 中 backbone 学习率相对 base lr 的缩放")
    ap.add_argument("--head-lr-scale", type=float, default=1.0,
                    help="Stage 2 中 gain head 学习率相对 base lr 的缩放")
    ap.add_argument("--unfreeze-last-n-layers", type=int, default=0,
                    help="仅解冻 transformer backbone 最后 N 层；0 表示全 backbone")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup = prepare_feature_setup(
        transitions_path=args.transitions,
        hierarchy_path=args.hierarchy,
        no_meta=args.no_meta,
        meta_fields=args.meta_fields,
    )
    gain_trans = filter_gain_transitions(setup.transitions)

    model = TwoStageTransitionModel(
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
        seed=args.seed,
        feat_meta=setup.feat_meta,
    )
    pretrained_source = None
    if args.pretrained:
        model.load_pretrained(args.pretrained)
        pretrained_source = os.path.abspath(args.pretrained)

    fit_report = model.fit_transitions(
        gain_trans,
        epochs=args.epochs,
        pretrained=bool(args.pretrained),
        freeze_backbone=bool(args.freeze_backbone),
        freeze_backbone_epochs=int(args.freeze_backbone_epochs),
        backbone_lr_scale=float(args.backbone_lr_scale),
        head_lr_scale=float(args.head_lr_scale),
        unfreeze_last_n_layers=int(args.unfreeze_last_n_layers),
    )
    model_path = os.path.join(args.outdir, "model.pt")
    model.save(
        model_path,
        extra={
            "feature_setup": {
                "transition_input": _jsonable(setup.transition_info),
                "feat_meta": _jsonable(setup.meta_info),
                "hierarchy_path": setup.hierarchy_path,
            },
            "pretrained_source": pretrained_source,
        },
    )

    rank_df = first_step_ranking_table(model, setup.feat_ids, setup.hierarchy_df)
    rank_df.to_csv(os.path.join(args.outdir, "first_step_ranking.csv"), index=False)

    payload = {
        "transitions": os.path.abspath(args.transitions),
        "model_path": os.path.abspath(model_path),
        "pretrained_source": pretrained_source,
        "feature_setup": {
            "transition_input": _jsonable(setup.transition_info),
            "feat_meta": _jsonable(setup.meta_info),
            "hierarchy_path": setup.hierarchy_path,
        },
        "n_gain_transitions": int(len(gain_trans)),
        "pretrain_report": _jsonable(model.pretrain_report),
        "stage2_fit_report": _jsonable(fit_report),
    }
    with open(os.path.join(args.outdir, "stage2_fit_report.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[stage2] transitions={len(gain_trans)} L={setup.L} arch={args.arch}")
    print(f"[stage2] pretrained: {pretrained_source or '<random_init>'}")
    print(f"[stage2] model     : {model_path}")
    print(f"[stage2] report    : {os.path.join(args.outdir, 'stage2_fit_report.json')}")


if __name__ == "__main__":
    main()
