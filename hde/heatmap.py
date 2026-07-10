"""Top-down SURPRISE maps of real kitchens -- p(location | kitchen) over the (x,y)
floor plane. These reveal the MULTIMODAL structure (hob / sink / counters / storage)
that the Flow captures and a single Gaussian cannot.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import config as C
from .model import PlacementModel


def _load():
    df = pd.read_parquet(C.POINTS_PARQUET)
    vocab = json.loads(C.VOCAB_JSON.read_text()); norm = json.loads(C.NORM_JSON.read_text())
    ck = torch.load(C.MODEL_PT, map_location="cpu")
    flow = PlacementModel(ck["n_kitchen"], head="flow"); flow.load_state_dict(ck["state"]); flow.eval()
    return df, vocab, norm, flow


def _grid(flow, ki, norm, pts, res=70):
    xs = np.linspace(pts["x"].min() - 0.3, pts["x"].max() + 0.3, res)
    ys = np.linspace(pts["y"].min() - 0.3, pts["y"].max() + 0.3, res)
    zmed = float(pts["z"].median())
    gx, gy = np.meshgrid(xs, ys)
    g = np.stack([gx.ravel(), gy.ravel(), np.full(gx.size, zmed)], axis=1).astype("float32")
    for j, k in enumerate(C.COORDS):
        g[:, j] = (g[:, j] - norm[k][0]) / norm[k][1]
    kk = torch.full((g.shape[0],), ki, dtype=torch.long)
    with torch.no_grad():
        lp = flow.log_prob(kk, torch.tensor(g)).numpy()
    return xs, ys, (-lp).reshape(res, res)


def main():
    df, vocab, norm, flow = _load()
    kitchens = sorted(df["kitchen"].unique())[:6]
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for ax, kit in zip(axes.ravel(), kitchens):
        pts = df[df["kitchen"] == kit]
        xs, ys, S = _grid(flow, vocab["kitchen"][kit], norm, pts)
        ax.contourf(xs, ys, S, levels=20, cmap="viridis_r")
        sub = pts.sample(min(400, len(pts)), random_state=0)
        ax.scatter(sub["x"], sub["y"], s=6, c="white", edgecolor="k", lw=0.2, alpha=0.6)
        ax.set_title(f"{kit}  ({len(pts)} interactions)", fontsize=11)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    for ax in axes.ravel()[len(kitchens):]:
        ax.axis("off")
    fig.suptitle("SURPRISE = -log p(location | kitchen)  — dark = surprising · "
                 "white = real interactions (note the multimodal hotspots)", fontsize=13)
    fig.tight_layout(); fig.savefig(C.FIGS / "surprise_maps.png", dpi=110); plt.close(fig)
    print("wrote surprise_maps.png", kitchens)


if __name__ == "__main__":
    main()
