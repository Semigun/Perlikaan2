"""Head-to-head of two bot files (e.g. new vs snapshot) at 7-player tables."""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.arena import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="perudo/bots/orakel.py")
    ap.add_argument("--b", default="perudo/bots/orakel.py")
    ap.add_argument("--pa", default="{}")
    ap.add_argument("--pb", default="{}")
    ap.add_argument("--games", type=int, default=150)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    pa, pb = json.loads(a.pa), json.loads(a.pb)

    def A(i):
        return ("bot", a.a, "OrakelBot", pa, f"A{i}")

    def B(i):
        return ("bot", a.b, "OrakelBot", pb, f"B{i}")

    def z(ks):
        return [("zoo", k, f"{k}#{i}") for i, k in enumerate(ks)]

    lineups = {
        "3v3+evo": [A(0), A(1), A(2), B(0), B(1), B(2)] + z(["evo1"]),
        "2v2+evo": [A(0), A(1), B(0), B(1)] + z(["evo1", "evo2", "evo3"]),
        "1v1+evo": [A(0), B(0)] + z(["evo1", "evo2", "evo3", "believer", "130"]),
    }
    t = time.time()
    tot = {"A": [0, 0], "B": [0, 0]}
    for name, lu in lineups.items():
        if a.only and name not in a.only:
            continue
        st = evaluate([lu] * a.chunks, a.games, seed=zlib.crc32(name.encode()) % 100000 + 7919 * a.seed)
        row = {}
        for side in "AB":
            w = sum(v[0] for k, v in st.items() if k.startswith(side))
            g = sum(v[1] for k, v in st.items() if k.startswith(side))
            tot[side][0] += w
            tot[side][1] += g
            row[side] = 100 * w / g
        print(f"{name:<10} A {row['A']:5.1f}%  B {row['B']:5.1f}%  (per seat; fair {100 / 7:.1f}%)", flush=True)
    ta, tb = (100 * tot[s][0] / tot[s][1] for s in "AB")
    print(f"{'overall':<10} A {ta:5.1f}%  B {tb:5.1f}%   {time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
