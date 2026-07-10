"""Replay: glide a probe over a real kitchen's learned placement density and watch
SURPRISE change live. The probe visits two typical spots (low surprise) then an
unusual spot (high surprise) -- "if you leave it HERE you'd probably forget it".
Output: GIF (+ mp4 if ffmpeg).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

from . import config as C
from .model import PlacementModel


def main():
    df = pd.read_parquet(C.POINTS_PARQUET)
    vocab = json.loads(C.VOCAB_JSON.read_text()); norm = json.loads(C.NORM_JSON.read_text())
    ck = torch.load(C.MODEL_PT, map_location="cpu")
    flow = PlacementModel(ck["n_kitchen"], head="flow"); flow.load_state_dict(ck["state"]); flow.eval()

    kit = df["kitchen"].value_counts().idxmax()
    pts = df[df["kitchen"] == kit]
    ki = vocab["kitchen"][kit]
    zmed = float(pts["z"].median())

    def surprise_xy(xy):
        p = np.column_stack([xy[:, 0], xy[:, 1], np.full(len(xy), zmed)]).astype("float32")
        for j, k in enumerate(C.COORDS):
            p[:, j] = (p[:, j] - norm[k][0]) / norm[k][1]
        kk = torch.full((len(p),), ki, dtype=torch.long)
        with torch.no_grad():
            return (-flow.log_prob(kk, torch.tensor(p))).numpy()

    res = 80
    xs = np.linspace(pts["x"].min() - 0.2, pts["x"].max() + 0.2, res)
    ys = np.linspace(pts["y"].min() - 0.2, pts["y"].max() + 0.2, res)
    gx, gy = np.meshgrid(xs, ys)
    S = surprise_xy(np.column_stack([gx.ravel(), gy.ravel()])).reshape(res, res)
    vmin, vmax = np.percentile(S, 3), np.percentile(S, 97)
    flat = S.ravel()
    lo1 = np.array([gx.ravel()[flat.argmin()], gy.ravel()[flat.argmin()]])
    far = (gx.ravel() - lo1[0]) ** 2 + (gy.ravel() - lo1[1]) ** 2
    masked = np.where(far > (0.35 * np.ptp(xs)) ** 2, flat, flat.max())
    lo2 = np.array([gx.ravel()[masked.argmin()], gy.ravel()[masked.argmin()]])
    hi = np.array([gx.ravel()[flat.argmax()], gy.ravel()[flat.argmax()]])

    def leg(a, b, n):
        return [a + (b - a) * t for t in np.linspace(0, 1, n, endpoint=False)]
    path = np.array(leg(lo1, lo2, 30) + leg(lo2, hi, 34) + leg(hi, lo1, 30))
    ps = surprise_xy(path)
    can = surprise_xy(pts[["x", "y"]].values)
    thr1, thr2 = np.percentile(can, 75), np.percentile(can, 98)

    def band(s):
        if s <= thr1:
            return "expected here", "#2e7d32"
        if s <= thr2:
            return "a bit unusual", "#ef6c00"
        return "very surprising", "#c62828"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 5), gridspec_kw={"width_ratios": [3, 1]})
    axL.contourf(xs, ys, S, levels=22, cmap="viridis_r", vmin=vmin, vmax=vmax)
    sub = pts.sample(min(350, len(pts)), random_state=0)
    axL.scatter(sub["x"], sub["y"], s=8, c="white", edgecolor="k", lw=0.2, alpha=0.6)
    trail, = axL.plot([], [], "-", c="white", lw=1.0, alpha=0.6)
    probe = axL.scatter([], [], s=260, edgecolor="k", lw=1.2, zorder=5)
    axL.set_title(f"Kitchen {kit}: sliding an object over the room")
    axL.set_xlabel("x [m]"); axL.set_ylabel("y [m]")

    axR.set_xlim(0, 1); axR.set_ylim(vmin, vmax); axR.set_xticks([])
    axR.set_title("SURPRISE gauge\n(-log p)")
    gauge = axR.bar(0.5, vmin, width=0.6, color="#2e7d32")[0]
    axR.axhline(thr1, ls="--", c="gray", lw=0.8); axR.axhline(thr2, ls="--", c="gray", lw=0.8)
    label = axR.text(0.5, vmin, "", ha="center", va="bottom", fontsize=11, fontweight="bold")
    from matplotlib.colors import Normalize
    cmap = plt.get_cmap("viridis_r"); cn = Normalize(vmin, vmax)

    def update(i):
        p = path[i]; s = ps[i]
        probe.set_offsets([p]); probe.set_color(cmap(cn(s)))
        trail.set_data(path[:i + 1, 0], path[:i + 1, 1])
        txt, col = band(s)
        gauge.set_height(min(max(s, vmin), vmax) - vmin); gauge.set_y(vmin); gauge.set_color(col)
        label.set_text(txt); label.set_color(col); label.set_y(min(max(s, vmin), vmax))
        return probe, trail, gauge, label

    anim = FuncAnimation(fig, update, frames=len(path), interval=70, blit=False)
    fig.tight_layout()
    anim.save(C.FIGS / "replay.gif", writer=PillowWriter(fps=15), dpi=90)
    print("wrote replay.gif", kit)
    try:
        anim.save(C.RESULTS / "replay.mp4", writer="ffmpeg", fps=15, dpi=110); print("wrote replay.mp4")
    except Exception as e:
        print("mp4 skipped:", e)
    plt.close(fig)


if __name__ == "__main__":
    main()
