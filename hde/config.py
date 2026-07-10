"""Paths, constants, and the HD-EPIC open-annotation URL."""
from __future__ import annotations

from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DATA = BASE / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = BASE / "results"
FIGS = RESULTS / "figures"
for _d in (RAW, PROC, RESULTS, FIGS):
    _d.mkdir(parents=True, exist_ok=True)

# HD-EPIC annotations are fully open on GitHub (no login, no video needed).
PRIMING_URL = ("https://raw.githubusercontent.com/hd-epic/hd-epic-annotations/"
               "main/eye-gaze-priming/priming_info.json")

COORDS = ["x", "y", "z"]
VAL_MOD = 5              # every 5th video (by stable hash) -> held-out

PRIMING_JSON = RAW / "priming_info.json"
POINTS_PARQUET = PROC / "points.parquet"
VOCAB_JSON = PROC / "vocab.json"
NORM_JSON = PROC / "norm_stats.json"
MODEL_PT = RESULTS / "model.pt"
METRICS_JSON = RESULTS / "metrics.json"
