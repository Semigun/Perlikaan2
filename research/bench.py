"""
Benchmark suite: score Orakel parameter variants on a fixed set of lineups
with common seeds.

Score = mean over lineups of (win rate x players at the table), so 1.0
means "fair share" and higher is better, regardless of table size.

    python research/bench.py --variants '{"base": {}, "k1": {"kappa": 1.0}}'
"""

from __future__ import annotations

import argparse
import json
import math
import zlib
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.arena import play_chunk  # noqa: E402

BOT = "perudo/bots/orakel.py"


def suite(params, chunks, pop_seed=0, bot=None, ref_params=None):
    bot = bot or BOT
    me = ("bot", bot, "OrakelBot", params, "Orakel")

    def z(keys):
        return [("zoo", k, f"{k}#{i}") for i, k in enumerate(keys)]

    def ref(i):
        return ("bot", bot, "OrakelBot", ref_params or {}, f"Ref{i}")

    def var(i, params):
        return ("bot", bot, "OrakelBot", dict(ref_params or {}, **params), f"Var{i}")

    # The tournament is a 7-player table of (presumably) strong bots, so
    # every lineup has 7 seats and most are full of strong opponents.
    lineups = {
        "evo": [me] + z(["evo1", "evo2", "evo3", "evo1", "believer", "evo2"]),
        "mirror": [me] + [ref(i) for i in range(6)],
        "mixstrong": [me, ref(0), ref(1)] + z(["evo1", "evo2", "evo3", "believer"]),
        "variants": [me, ref(0), var(1, {"learn": False}), var(2, {"gamma": 3.0}),
                     var(3, {"kappa": 1.2})] + z(["evo1", "evo3"]),
        "friends": [me] + z(["100", "110", "120", "130", "autist", "believer"]),
    }
    out = {}
    for name, lu in lineups.items():
        out[name] = [lu] * chunks
    # random populations: a different population per chunk (same for every variant)
    out["pop"] = [
        [me] + [("pop", pop_seed * 1000 + c * 10 + i, f"pop{i}") for i in range(6)]
        for c in range(chunks)
    ]
    return out


def run_variants(variants, games, chunks, seed, workers=None, only=None, bot=None, ref_params=None):
    jobs = []
    index = []
    for vname, params in variants.items():
        for lname, lineups in suite(params, chunks, bot=bot, ref_params=ref_params).items():
            if only and lname not in only:
                continue
            for c, lu in enumerate(lineups):
                jobs.append((lu, games, seed * 7919 + zlib.crc32(lname.encode()) % 1000 * 31 + c))
                index.append((vname, lname, len(lu)))
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for (vname, lname, size), (wins, _pos, g) in zip(index, ex.map(play_chunk, jobs)):
            w, n, sz = results.get((vname, lname), (0, 0, size))
            results[(vname, lname)] = (w + wins.get("Orakel", 0), n + g, sz)
    return results


def summarize(results, variants):
    lnames = sorted({l for (_, l) in results})
    print(f"{'variant':<16}" + "".join(f"{l:>9}" for l in lnames) + f"{'SCORE':>9}{'±2se':>7}")
    scores = {}
    for v in variants:
        row = []
        vals = []
        var = 0.0
        for l in lnames:
            w, n, sz = results[(v, l)]
            p = w / n
            row.append(f"{100 * p:>8.1f}%")
            vals.append(p * sz)
            var += (sz ** 2) * p * (1 - p) / n
        score = sum(vals) / len(vals)
        se = math.sqrt(var) / len(vals)
        scores[v] = score
        print(f"{v:<16}" + "".join(row) + f"{score:>9.3f}{2 * se:>7.3f}")
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default='{"base": {}}')
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()
    variants = json.loads(a.variants)
    t = time.time()
    res = run_variants(variants, a.games, a.chunks, a.seed, only=a.only)
    summarize(res, variants)
    print(f"{time.time() - t:.0f}s")


if __name__ == "__main__":
    main()
