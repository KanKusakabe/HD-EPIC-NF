"""Download HD-EPIC eye-gaze-priming annotations (3-D object locations + gaze). Open, no login."""
from __future__ import annotations

import urllib.request

from . import config as C


def main() -> None:
    out = C.PRIMING_JSON
    if out.exists() and out.stat().st_size > 0:
        print("skip (exists)", out.name)
        return
    print("download", C.PRIMING_URL)
    urllib.request.urlretrieve(C.PRIMING_URL, out)
    print("  ->", out.name, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
