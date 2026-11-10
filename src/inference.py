"""Use released EvoState checkpoints without retraining.

    import sys; sys.path.insert(0, "src")
    from inference import load_model

    m = load_model("tuberculosis_genes19")          # name from checkpoints/MANIFEST.json, or a path
    m.rank(present=["katG"], top_k=5)               # most likely next acquisitions
    p = m.predict_proba(present=["katG", "RRDR"])   # probability vector over all features

Checkpoints are loaded with ``torch.load(..., weights_only=True)``, which does not execute
arbitrary pickled code. The network is the unmodified ``TwoStageTransitionModel`` from
``src/models/two_stage_backbone.py``.

Probabilities are computed over *candidate* features: features absent from the given state,
excluding features marked as non-candidates in the checkpoint (the genetic-background markers of
the E. coli models, which are present from the start and are never acquired). Passing
``include_non_candidates=True`` returns the unrestricted distribution over all absent features,
which is the definition used for the cross-validation metrics in the manuscript. For the TB and
K. pneumoniae models the two definitions are identical.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from difflib import get_close_matches
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SRC_DIR)
CHECKPOINT_DIR = os.path.join(REPO_ROOT, "checkpoints")
MANIFEST_PATH = os.path.join(CHECKPOINT_DIR, "MANIFEST.json")

_MODELS_DIR = os.path.join(SRC_DIR, "models")
if _MODELS_DIR not in sys.path:
    sys.path.insert(0, _MODELS_DIR)

import two_stage_backbone as TSB  # noqa: E402

# Harmless PyTorch notice emitted when a TransformerEncoder is built without the nested-tensor path.
warnings.filterwarnings("ignore", message="enable_nested_tensor is True")


def list_models() -> Dict[str, dict]:
    """Return the released checkpoints described in checkpoints/MANIFEST.json."""
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)["models"]


def resolve_checkpoint(name_or_path: str) -> str:
    """Map a manifest name to its file; existing file paths are returned unchanged."""
    if os.path.isfile(name_or_path):
        return os.path.abspath(name_or_path)
    models = list_models()
    if name_or_path not in models:
        hint = [m for m in models if name_or_path.lower() in m.lower()][:5] \
            or get_close_matches(name_or_path, list(models), n=3)
        raise KeyError(f"unknown model {name_or_path!r}"
                       + (f"; did you mean {', '.join(hint)}?" if hint else "")
                       + " (run scripts/predict.py --list-models)")
    path = os.path.realpath(os.path.join(REPO_ROOT, models[name_or_path]["file"]))
    if not path.startswith(os.path.realpath(CHECKPOINT_DIR) + os.sep):
        raise ValueError("manifest entry points outside checkpoints/")
    return path


class EvoStateModel:
    """A loaded checkpoint plus its ordered feature names."""

    def __init__(self, net: "TSB.TwoStageTransitionModel", feature_names: Sequence[str],
                 non_candidate_features: Iterable[str] = (), release: Optional[dict] = None):
        self.net = net
        self.feature_names: List[str] = list(feature_names)
        self.index = {f: i for i, f in enumerate(self.feature_names)}
        self.non_candidates = np.zeros(len(self.feature_names), dtype=bool)
        for f in non_candidate_features:
            self.non_candidates[self.index[f]] = True
        self.release = release or {}
        if len(self.feature_names) != int(net.config["L"]):
            raise ValueError("feature list does not match the model input size")

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def encode(self, present: Iterable[str] = ()) -> np.ndarray:
        """Binary state vector with the named features set to 1."""
        state = np.zeros(self.n_features, dtype=np.int8)
        for f in present:
            if f not in self.index:
                hint = get_close_matches(str(f), self.feature_names, n=3)
                raise KeyError(f"unknown feature {f!r}"
                               + (f"; did you mean {', '.join(hint)}?" if hint else ""))
            state[self.index[f]] = 1
        return state

    def to_state(self, state=None, present=None) -> np.ndarray:
        """Accept a feature-name list, a 0/1 string or a 0/1 vector; return an int8 vector."""
        if state is None:
            return self.encode(present or ())
        if isinstance(state, str):
            if len(state) != self.n_features or set(state) - {"0", "1"}:
                raise ValueError(f"state string must be {self.n_features} characters of 0/1")
            return np.fromiter((c == "1" for c in state), dtype=np.int8, count=self.n_features)
        arr = np.asarray(state).astype(np.int8).reshape(-1)
        if arr.size != self.n_features or set(np.unique(arr).tolist()) - {0, 1}:
            raise ValueError(f"state must be a 0/1 vector of length {self.n_features}")
        return arr

    def candidates(self, state, include_non_candidates: bool = False) -> np.ndarray:
        cand = np.asarray(state) == 0
        if not include_non_candidates:
            cand &= ~self.non_candidates
        return cand

    def predict_proba(self, state=None, present=None, include_non_candidates: bool = False) -> np.ndarray:
        """Probability of each feature being the next acquisition (0 for non-candidates).

        The softmax scores are renormalised over the candidates, as in
        ``src/eval/evaluator.py::_normalize_on_candidates``, so the result sums to 1.
        """
        s = self.to_state(state, present)
        p = self.net.predict_scores(s)
        cand = self.candidates(s, include_non_candidates)
        p = np.where(cand, p, 0.0)
        total = p.sum()
        if total > 0:
            return p / total
        if cand.any():
            p[cand] = 1.0 / cand.sum()
        return p

    def rank(self, state=None, present=None, top_k: Optional[int] = 10,
             include_non_candidates: bool = False) -> List[dict]:
        """Candidate features sorted by predicted probability."""
        s = self.to_state(state, present)
        p = self.predict_proba(s, include_non_candidates=include_non_candidates)
        cand = self.candidates(s, include_non_candidates)
        order = [int(i) for i in np.argsort(-p, kind="stable") if cand[i]]
        if top_k is not None:
            order = order[:top_k]
        return [{"rank": r + 1, "feature": self.feature_names[i], "feature_index": i,
                 "probability": float(p[i])} for r, i in enumerate(order)]


def load_model(name_or_path: str, device: str = "cpu") -> EvoStateModel:
    """Load a released checkpoint by manifest name or file path."""
    path = resolve_checkpoint(name_or_path)
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if "feature_names" not in blob:
        raise ValueError("checkpoint has no embedded feature_names; use a file from checkpoints/")
    loss_mask = blob.get("loss_mask")
    if loss_mask is not None:
        loss_mask = np.asarray(loss_mask, dtype=np.float32)
    head_mode = blob.get("head_mode", "gain_only")
    net = TSB.TwoStageTransitionModel(
        feat_meta=TSB.deserialize_feat_meta(blob.get("feat_meta")),
        device=device, head_mode=head_mode, loss_mask=loss_mask,
        loss_weight=float(blob.get("loss_weight", 1.0)), **dict(blob["config"]),
    )
    net.head_mode = blob.get("requested_head_mode", head_mode)
    net._effective_head_mode = head_mode
    result = net.net.load_state_dict(blob["state_dict"], strict=False)
    # Older checkpoints predate the (unused) loss head; anything else must match exactly.
    bad_missing = [k for k in result.missing_keys if not k.startswith("loss_head.")]
    if result.unexpected_keys or bad_missing or (result.missing_keys and head_mode != "gain_only"):
        raise ValueError(f"checkpoint/model mismatch: missing={result.missing_keys} "
                         f"unexpected={result.unexpected_keys}")
    net.net.eval()
    net.pretrain_report = blob.get("pretrain_report") or {}
    net.fit_report = blob.get("fit_report") or {}
    return EvoStateModel(net, blob["feature_names"], blob.get("non_candidate_features") or [],
                         blob.get("release"))
