"""Kitchen vocab + coordinate normalisation (train split only)."""
from __future__ import annotations

import json

import pandas as pd

from . import config as C


def main() -> None:
    df = pd.read_parquet(C.POINTS_PARQUET)
    tr = df[df["split"] == "train"]
    vocab = {"kitchen": {k: i for i, k in enumerate(sorted(df["kitchen"].unique()))}}
    C.VOCAB_JSON.write_text(json.dumps(vocab, indent=1))
    norm = {k: [float(tr[k].mean()), float(tr[k].std() + 1e-6)] for k in C.COORDS}
    C.NORM_JSON.write_text(json.dumps(norm, indent=1))
    print("kitchens", len(vocab["kitchen"]), "norm", norm)


if __name__ == "__main__":
    main()
