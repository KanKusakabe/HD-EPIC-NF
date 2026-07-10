"""Deep study L1-L4: from an anomaly *detector* to a model of an individual's
routine and its usefulness for intervention.

  L1 personal vs population -- does knowing WHO improves placement prediction?
  L2 temporal drift         -- does a routine learned early still fit later?
  L3 few-shot transfer      -- how fast can a new kitchen be personalised from a prior?
  L4 predict vs reactive     -- is acting on surprise BEFORE the fact worth it?

All run on HD-EPIC open annotations, reusing the Experiment-B Flow. Results feed
results/metrics.json under keys l1..l4 and figures results/figures/l{1..4}_*.png.
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

torch.manual_seed(0)
np.random.seed(0)


def device():
    return "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")


def _load():
    df = pd.read_parquet(C.POINTS_PARQUET)
    vocab = json.loads(C.VOCAB_JSON.read_text())
    norm = json.loads(C.NORM_JSON.read_text())
    for k in C.COORDS:
        m, s = norm[k]
        df[k] = (df[k] - m) / s
    df["ki"] = df["kitchen"].map(vocab["kitchen"]).astype(int)
    df["date"] = df["video"].str.split("-").str[1]
    return df, vocab


def _train(X, dev, epochs=50, lr=1e-3, batch=2048, init_state=None, val_X=None, verbose=False):
    """Train an UNCONDITIONED NSF on points X (Nx3). Returns (best_state, best_nll)."""
    m = PlacementModel(1, head="flow").to(dev)
    if init_state is not None:
        m.load_state_dict(init_state)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    X = X.to(dev); n = X.shape[0]
    z = torch.zeros(n, dtype=torch.long, device=dev)
    vz = None
    if val_X is not None:
        val_X = val_X.to(dev); vz = torch.zeros(val_X.shape[0], dtype=torch.long, device=dev)
    best, best_state = float("inf"), None
    for ep in range(epochs):
        m.train(); perm = torch.randperm(n, device=dev)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            loss = m.nll(z[idx], X[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            cur = float(m.nll(vz, val_X)) if val_X is not None else float(m.nll(z, X))
        if cur < best:
            best = cur; best_state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return best_state, best


@torch.no_grad()
def _nll_uncond(state, X, dev):
    m = PlacementModel(1, head="flow").to(dev); m.load_state_dict(state); m.eval()
    z = torch.zeros(X.shape[0], dtype=torch.long, device=dev)
    return float(m.nll(z, X.to(dev)))


# ---------------------------------------------------------------- L1
def l1_personalization(df, vocab, dev):
    ck = torch.load(C.MODEL_PT, map_location=dev)
    personal = PlacementModel(ck["n_kitchen"], head="flow").to(dev)
    personal.load_state_dict(ck["state"]); personal.eval()

    tr = df[df["split"] == "train"]
    pop_state, _ = _train(torch.tensor(tr[C.COORDS].values, dtype=torch.float32), dev, epochs=60)
    pop = PlacementModel(1, head="flow").to(dev); pop.load_state_dict(pop_state); pop.eval()

    va = df[df["split"] == "val"]
    rows = []
    for kit, g in va.groupby("kitchen"):
        X = torch.tensor(g[C.COORDS].values, dtype=torch.float32, device=dev)
        ki = torch.full((len(g),), vocab["kitchen"][kit], dtype=torch.long, device=dev)
        z = torch.zeros(len(g), dtype=torch.long, device=dev)
        with torch.no_grad():
            nll_p = float(-personal.log_prob(ki, X).mean())
            nll_pop = float(-pop.log_prob(z, X).mean())
        rows.append((kit, nll_pop, nll_p, nll_pop - nll_p))
    r = pd.DataFrame(rows, columns=["kitchen", "nll_pop", "nll_personal", "gain"])
    gain = float(r["gain"].mean())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(r))
    ax.bar(x - 0.2, r["nll_pop"], 0.4, label="population model", color="#8b93a1")
    ax.bar(x + 0.2, r["nll_personal"], 0.4, label="personal (kitchen-conditioned)", color="#d97757")
    ax.set_xticks(x); ax.set_xticklabels(r["kitchen"]); ax.set_ylabel("held-out NLL (lower=better)")
    ax.set_title(f"L1 · personal vs population  (mean gain {gain:.2f} nats)"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIGS / "l1_personalization.png", dpi=110); plt.close(fig)
    return {"mean_gain_nats": round(gain, 3),
            "per_kitchen_gain": {k: round(v, 3) for k, v in zip(r["kitchen"], r["gain"])}}


# ---------------------------------------------------------------- L2
def l2_drift(df, dev):
    rows = []
    for kit, g in df.groupby("kitchen"):
        g = g.sort_values(["date", "video"])
        if g["date"].nunique() < 2 or len(g) < 300:
            continue
        X = torch.tensor(g[C.COORDS].values, dtype=torch.float32)
        cut = int(0.7 * len(g))
        Xtr = X[:cut]
        fut = X[cut:]                                   # future events (later days)
        ridx = np.random.permutation(len(g))[:len(fut)]
        rnd = X[ridx]                                   # random held-out (time-agnostic)
        state, _ = _train(Xtr, dev, epochs=60)
        nll_fut = _nll_uncond(state, fut, dev)
        nll_rnd = _nll_uncond(state, rnd, dev)
        rows.append((kit, nll_rnd, nll_fut, nll_fut - nll_rnd))
    r = pd.DataFrame(rows, columns=["kitchen", "nll_random", "nll_future", "drift"])
    drift = float(r["drift"].mean())

    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(r))
    ax.bar(x - 0.2, r["nll_random"], 0.4, label="random held-out", color="#8b93a1")
    ax.bar(x + 0.2, r["nll_future"], 0.4, label="future days (held-out in time)", color="#d97757")
    ax.set_xticks(x); ax.set_xticklabels(r["kitchen"]); ax.set_ylabel("held-out NLL")
    ax.set_title(f"L2 · does an early routine still fit later days?  (mean drift {drift:+.2f} nats)"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIGS / "l2_drift.png", dpi=110); plt.close(fig)
    return {"mean_drift_nats": round(drift, 3),
            "per_kitchen_drift": {k: round(v, 3) for k, v in zip(r["kitchen"], r["drift"])}}


# ---------------------------------------------------------------- L3
def l3_fewshot(df, dev, shots=(20, 50, 100, 200, 400)):
    kitchens = sorted(df["kitchen"].unique())
    pre_curve = {k: [] for k in shots}
    scr_curve = {k: [] for k in shots}
    for target in kitchens:
        others = df[(df["kitchen"] != target) & (df["split"] == "train")]
        pre_state, _ = _train(torch.tensor(others[C.COORDS].values, dtype=torch.float32), dev, epochs=40)
        tgt = df[df["kitchen"] == target].sample(frac=1.0, random_state=0)
        n_test = min(500, len(tgt) // 3)
        test = torch.tensor(tgt[C.COORDS].values[:n_test], dtype=torch.float32)
        pool = tgt[C.COORDS].values[n_test:]
        for k in shots:
            if len(pool) < k:
                continue
            Xk = torch.tensor(pool[:k], dtype=torch.float32)
            ft, nf = _train(Xk, dev, epochs=80, lr=5e-4, init_state=pre_state, val_X=test)
            sc, ns = _train(Xk, dev, epochs=120, lr=1e-3, val_X=test)
            pre_curve[k].append(nf); scr_curve[k].append(ns)
    pre = {k: float(np.mean(v)) for k, v in pre_curve.items() if v}
    scr = {k: float(np.mean(v)) for k, v in scr_curve.items() if v}

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    ks = sorted(pre)
    ax.plot(ks, [pre[k] for k in ks], "-o", label="pretrained on 8 kitchens + few-shot", color="#d97757")
    ax.plot(ks, [scr[k] for k in ks], "-o", label="from scratch on target only", color="#8b93a1")
    ax.set_xlabel("# adaptation examples from the new kitchen"); ax.set_ylabel("held-out NLL")
    ax.set_xscale("log"); ax.set_title("L3 · personalising a new kitchen from a public prior"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIGS / "l3_fewshot.png", dpi=110); plt.close(fig)
    # shots for pretrained to beat scratch's best-with-400
    return {"pretrained_nll": {str(k): round(v, 3) for k, v in pre.items()},
            "scratch_nll": {str(k): round(v, 3) for k, v in scr.items()}}


# ---------------------------------------------------------------- L4
def l4_policy(df, vocab, dev, benefit=1.0, fa_cost=0.15):
    ck = torch.load(C.MODEL_PT, map_location=dev)
    flow = PlacementModel(ck["n_kitchen"], head="flow").to(dev); flow.load_state_dict(ck["state"]); flow.eval()
    va = df[df["split"] == "val"].reset_index(drop=True)
    bbox = df.groupby("kitchen")[C.COORDS].agg(["min", "max"])
    rng = np.random.default_rng(0)
    inj = va.copy()
    for k in C.COORDS:
        lo = inj["kitchen"].map(bbox[(k, "min")]).values; hi = inj["kitchen"].map(bbox[(k, "max")]).values
        inj[k] = lo + rng.random(len(inj)) * (hi - lo)

    def surprise(frame):
        ki = torch.tensor(frame["kitchen"].map(vocab["kitchen"]).values, dtype=torch.long, device=dev)
        X = torch.tensor(frame[C.COORDS].values, dtype=torch.float32, device=dev)
        with torch.no_grad():
            return (-flow.log_prob(ki, X)).cpu().numpy()

    s_norm = surprise(va); s_risk = surprise(inj)          # at-risk = misplaced object
    thr = np.linspace(min(s_norm.min(), s_risk.min()), max(s_norm.max(), s_risk.max()), 200)
    n = len(va)
    # predictive: alert when surprise>thr. catch = at-risk alerted (benefit); FP = normal alerted (cost)
    catch = np.array([(s_risk > t).mean() for t in thr])
    fp = np.array([(s_norm > t).mean() for t in thr])
    util_pred = benefit * catch - fa_cost * fp
    best = int(np.argmax(util_pred))
    # reactive: never alert -> catches nothing, pays no false alarm -> utility 0
    util_reactive = 0.0

    fig, ax = plt.subplots(figsize=(7.8, 4.4))
    ax.plot(thr, util_pred, color="#d97757", label="predictive (alert on surprise)")
    ax.axhline(util_reactive, ls="--", color="#8b93a1", label="reactive (act only after)")
    ax.scatter([thr[best]], [util_pred[best]], color="#c2410c", zorder=5)
    ax.annotate(f"best {util_pred[best]:.2f}\n(catch {catch[best]:.0%}, false-alarm {fp[best]:.0%})",
                (thr[best], util_pred[best]), textcoords="offset points", xytext=(8, -6), fontsize=9)
    ax.set_xlabel("surprise threshold"); ax.set_ylabel(f"expected utility (benefit={benefit}, FA cost={fa_cost})")
    ax.set_title("L4 · predict-vs-reactive misplacement alerts"); ax.legend()
    fig.tight_layout(); fig.savefig(C.FIGS / "l4_policy.png", dpi=110); plt.close(fig)
    return {"best_predictive_utility": round(float(util_pred[best]), 3),
            "reactive_utility": 0.0,
            "catch_at_best": round(float(catch[best]), 3),
            "false_alarm_at_best": round(float(fp[best]), 3),
            "benefit": benefit, "fa_cost": fa_cost}


def main():
    dev = device(); print("device:", dev)
    df, vocab = _load()
    out = {}
    print("L1 personalization..."); out["l1"] = l1_personalization(df, vocab, dev); print("  ", out["l1"]["mean_gain_nats"])
    print("L2 drift...");           out["l2"] = l2_drift(df, dev); print("  ", out["l2"]["mean_drift_nats"])
    print("L3 few-shot...");        out["l3"] = l3_fewshot(df, dev); print("  ", out["l3"]["pretrained_nll"])
    print("L4 policy...");          out["l4"] = l4_policy(df, vocab, dev); print("  ", out["l4"]["best_predictive_utility"])
    prev = json.loads(C.METRICS_JSON.read_text()) if C.METRICS_JSON.exists() else {}
    prev["deep"] = out
    C.METRICS_JSON.write_text(json.dumps(prev, indent=1))
    print("wrote deep metrics + figures")


if __name__ == "__main__":
    main()
