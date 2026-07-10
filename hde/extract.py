"""HD-EPIC eye-gaze-priming -> tidy parquet of object interaction locations.

Each priming event has a `start` and `end`, each carrying the object's 3-D
location and the gaze point at priming time. We pool both as interaction
locations, keeping the gaze offset (||location - gaze||) as a distraction signal.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from . import config as C


def _split(video: str) -> str:
    h = int(hashlib.md5(video.encode()).hexdigest(), 16)
    return "val" if h % C.VAL_MOD == 0 else "train"


def main() -> None:
    d = json.loads(C.PRIMING_JSON.read_text())
    rows = []
    for video, evs in d.items():
        if not isinstance(evs, dict):
            continue
        kitchen = video.split("-")[0]
        split = _split(video)
        for _, e in evs.items():
            if not isinstance(e, dict):
                continue
            for phase in ("start", "end"):
                p = e.get(phase)
                if not isinstance(p, dict):
                    continue
                loc = p.get("3d_location")
                if not (isinstance(loc, list) and len(loc) == 3):
                    continue
                ps = p.get("prime_stats", {}) or {}
                g = ps.get("gaze_point")
                goff = (float(np.linalg.norm(np.array(loc) - np.array(g)))
                        if isinstance(g, list) and len(g) == 3 else np.nan)
                rows.append(dict(
                    video=video, kitchen=kitchen, split=split, phase=phase,
                    x=loc[0], y=loc[1], z=loc[2], gaze_offset=goff,
                    dist_to_cam=ps.get("dist_to_cam", np.nan),
                    prime_gap=ps.get("prime_gap", np.nan)))
    df = pd.DataFrame(rows)
    df.to_parquet(C.POINTS_PARQUET)
    print(f"rows={len(df)}  kitchens={df['kitchen'].nunique()}  videos={df['video'].nunique()}  "
          f"splits={df['split'].value_counts().to_dict()}")
    print(f"with gaze={int(df['gaze_offset'].notna().sum())}  "
          f"gaze_offset median={df['gaze_offset'].median():.3f}")
    print("wrote", C.POINTS_PARQUET)


if __name__ == "__main__":
    main()
