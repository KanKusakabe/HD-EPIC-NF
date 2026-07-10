"""Does the Flow's multimodal density buy us anything over the GMM baseline?

  1. random-teleport detection -- AUC(surprise -> object dropped at a random spot
     in the kitchen). Flow's multimodality should win here.
  2. gaze coupling -- are more-surprising placements the ones made further from
     where the person was looking? (Spearman of surprise vs gaze offset.)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from . import config as C
from .model import PlacementModel

RNG = np.random.default_rng(0)


def auc(scores, labels):
    labels = np.asarray(labels).astype(int)
    n_pos, n_neg = int(labels.sum()), int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def spearman(a, b):
    a, b = np.asarray(a), np.asarray(b)
    ar = np.argsort(np.argsort(a)); br = np.argsort(np.argsort(b))
    ar = (ar - ar.mean()) / (ar.std() + 1e-9); br = (br - br.mean()) / (br.std() + 1e-9)
    return float((ar * br).mean())


@torch.no_grad()
def _surprise(model, ki, y, bs=8192):
    out = []
    for i in range(0, len(ki), bs):
        out.append((-model.log_prob(ki[i:i + bs], y[i:i + bs])).numpy())
    return np.concatenate(out)


def main() -> None:
    df = pd.read_parquet(C.POINTS_PARQUET)
    vocab = json.loads(C.VOCAB_JSON.read_text()); norm = json.loads(C.NORM_JSON.read_text())
    for k in C.COORDS:
        m, s = norm[k]; df[k] = (df[k] - m) / s
    df["ki"] = df["kitchen"].map(vocab["kitchen"]).astype(int)

    ck = torch.load(C.MODEL_PT, map_location="cpu")
    flow = PlacementModel(ck["n_kitchen"], head="flow"); flow.load_state_dict(ck["state"]); flow.eval()
    gmm = PlacementModel(ck["n_kitchen"], head="gmm"); gmm.load_state_dict(ck["gmm_state"]); gmm.eval()

    va = df[df["split"] == "val"].reset_index(drop=True)
    ki = torch.tensor(va["ki"].values, dtype=torch.long)
    y = torch.tensor(va[C.COORDS].values, dtype=torch.float32)
    s_flow = _surprise(flow, ki, y); s_gmm = _surprise(gmm, ki, y)

    # per-kitchen bbox (from all points) for random teleport injection
    bbox = df.groupby("kitchen")[C.COORDS].agg(["min", "max"])
    inj = va.copy()
    for k in C.COORDS:
        lo = inj["kitchen"].map(bbox[(k, "min")]).values
        hi = inj["kitchen"].map(bbox[(k, "max")]).values
        inj[k] = lo + RNG.random(len(inj)) * (hi - lo)
    yi = torch.tensor(inj[C.COORDS].values, dtype=torch.float32)
    si_flow = _surprise(flow, ki, yi); si_gmm = _surprise(gmm, ki, yi)
    lab = np.r_[np.zeros(len(va)), np.ones(len(inj))]

    res = {
        "injection_auc_flow": round(auc(np.r_[s_flow, si_flow], lab), 4),
        "injection_auc_gmm": round(auc(np.r_[s_gmm, si_gmm], lab), 4),
        "n_val": int(len(va)),
    }
    # gaze coupling on val points that have a gaze offset
    m = va["gaze_offset"].notna().values
    if m.sum() > 50:
        res["gaze_surprise_spearman_flow"] = round(spearman(s_flow[m], va["gaze_offset"].values[m]), 4)
        hi = va["gaze_offset"].values[m] > np.nanpercentile(va["gaze_offset"].values[m], 75)
        res["surprise_high_gazeoffset_mean"] = round(float(s_flow[m][hi].mean()), 3)
        res["surprise_low_gazeoffset_mean"] = round(float(s_flow[m][~hi].mean()), 3)

    prev = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else {}
    prev["evaluate"] = res
    C.METRICS_JSON.write_text(json.dumps(prev, indent=1))
    for k, v in res.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
