"""Train p(location | kitchen) on HD-EPIC interaction points; held-out = 20% of videos.

Trains the Flow AND the GMM baseline so the report can honestly show whether the
Flow's expressivity pays off on a genuinely multimodal kitchen layout.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from . import config as C
from .model import PlacementModel


def device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load(dev):
    df = pd.read_parquet(C.POINTS_PARQUET)
    vocab = json.loads(C.VOCAB_JSON.read_text())
    norm = json.loads(C.NORM_JSON.read_text())
    for k in C.COORDS:
        m, s = norm[k]
        df[k] = (df[k] - m) / s
    df["ki"] = df["kitchen"].map(vocab["kitchen"]).astype(int)

    def pack(split):
        d = df[df["split"] == split]
        return (torch.tensor(d["ki"].values, dtype=torch.long, device=dev),
                torch.tensor(d[C.COORDS].values, dtype=torch.float32, device=dev))

    return pack("train"), pack("val"), vocab


def _train(head, tr, va, n_kitchen, dev, epochs, batch, lr):
    model = PlacementModel(n_kitchen, head=head).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = tr[0].shape[0]
    hist, best, best_state = [], float("inf"), None
    for ep in range(epochs):
        model.train(); perm = torch.randperm(n, device=dev); tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = model.nll(tr[0][idx], tr[1][idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()) * len(idx)
        model.eval()
        with torch.no_grad():
            vnll = float(model.nll(*va))
        hist.append({"epoch": ep, "train_nll": tot / n, "val_nll": vnll})
        if vnll < best:
            best = vnll
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if ep % 8 == 0 or ep == epochs - 1:
            print(f"  [{head}] ep {ep:3d}  train {tot/n:7.3f}  val {vnll:7.3f}")
    model.load_state_dict(best_state)
    return model, {"history": hist, "best_val_nll": best}


def main(epochs=80, batch=2048, lr=1e-3, fast=False):
    dev = device(); print("device:", dev)
    if fast:
        epochs = 20
    tr, va, vocab = load(dev)
    nk = len(vocab["kitchen"])
    print(f"train {tr[0].shape[0]:,}  val {va[0].shape[0]:,}  kitchens {nk}")
    flow, mf = _train("flow", tr, va, nk, dev, epochs, batch, lr)
    gmm, mg = _train("gmm", tr, va, nk, dev, epochs, batch, lr)
    torch.save({"state": flow.state_dict(), "gmm_state": gmm.state_dict(), "n_kitchen": nk}, C.MODEL_PT)
    prev = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else {}
    prev["train"] = {"flow": mf, "gmm_baseline": mg,
                     "n_train": int(tr[0].shape[0]), "n_val": int(va[0].shape[0])}
    C.METRICS_JSON.write_text(json.dumps(prev, indent=1))
    print(f"Flow held-out NLL {mf['best_val_nll']:.3f} vs GMM {mg['best_val_nll']:.3f} -> saved {C.MODEL_PT.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--epochs", type=int, default=80)
    a = ap.parse_args()
    main(epochs=a.epochs, fast=a.fast)
