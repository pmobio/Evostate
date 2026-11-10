#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys


def _load_two_stage_backbone_module():
    here = os.path.dirname(os.path.abspath(__file__))
    module_path = os.path.join(here, "two_stage_backbone.py")
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
_jsonable = _TSB._jsonable
build_state_examples = _TSB.build_state_examples
prepare_feature_setup = _TSB.prepare_feature_setup


def main():
    ap = argparse.ArgumentParser(description="Stage 1 masked-state + contrastive pretraining")
    ap.add_argument("--transitions", required=True)
    ap.add_argument("--outdir", required=True)
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
    ap.add_argument("--no-dedupe", action="store_true")
    ap.add_argument("--hierarchy", default=None)
    ap.add_argument("--no-meta", action="store_true")
    ap.add_argument("--meta-fields", default="")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    setup = prepare_feature_setup(
        transitions_path=args.transitions,
        hierarchy_path=args.hierarchy,
        no_meta=args.no_meta,
        meta_fields=args.meta_fields,
    )
    states = build_state_examples(
        setup.transitions,
        state_source=args.state_source,
        dedupe=(not args.no_dedupe),
    )

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
    report = model.pretrain(
        states,
        epochs=args.epochs,
        mask_prob=args.mask_prob,
        mask_prob_pos=args.mask_prob_pos,
        mask_prob_neg=args.mask_prob_neg,
        contrastive_weight=args.contrastive_weight,
        contrastive_temperature=args.contrastive_temperature,
        contrastive_warmup_epochs=args.contrastive_warmup_epochs,
        similar_state_neg_threshold=args.similar_state_neg_threshold,
        checkpoint_metric=args.pretrain_checkpoint_metric,
    )

    ckpt_path = os.path.join(args.outdir, "stage1_model.pt")
    model.save(
        ckpt_path,
        extra={
            "feature_setup": {
                "transition_input": _jsonable(setup.transition_info),
                "feat_meta": _jsonable(setup.meta_info),
                "hierarchy_path": setup.hierarchy_path,
            },
            "state_dataset": {
                "state_source": args.state_source,
                "dedupe": bool(not args.no_dedupe),
                "n_states": int(len(states)),
                "n_features": int(setup.L),
            },
        },
    )

    payload = {
        "transitions": os.path.abspath(args.transitions),
        "checkpoint": os.path.abspath(ckpt_path),
        "feature_setup": {
            "transition_input": _jsonable(setup.transition_info),
            "feat_meta": _jsonable(setup.meta_info),
            "hierarchy_path": setup.hierarchy_path,
        },
        "state_dataset": {
            "state_source": args.state_source,
            "dedupe": bool(not args.no_dedupe),
            "n_states": int(len(states)),
            "n_features": int(setup.L),
        },
        "pretrain_report": _jsonable(report),
    }
    with open(os.path.join(args.outdir, "stage1_report.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"[stage1] states={len(states)} L={setup.L} arch={args.arch}")
    print(f"[stage1] checkpoint: {ckpt_path}")
    print(f"[stage1] report    : {os.path.join(args.outdir, 'stage1_report.json')}")


if __name__ == "__main__":
    main()
