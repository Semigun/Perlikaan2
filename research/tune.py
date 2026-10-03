"""
Coordinate-descent parameter tuning for Orakel on the benchmark suite.

Runs against a frozen snapshot of perudo/bots/orakel.py (so the live file
can keep changing) and appends every result to a log file.
"""
import argparse, json, shutil, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.bench import run_variants, summarize

GRID = {
    "explore_eps": [0.0, 0.001, 0.003],
    "gamma": [1.5, 2.5],
    "kappa": [0.5, 0.8, 1.2],
    "alpha_bid": [2.0, 6.0, 15.0],
    "beta_call": [1.0, 4.0, 10.0],
    "accept_pow": [0.0, 0.3],
    "max_steps": [2, 4, 6],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--passes", type=int, default=1)
    ap.add_argument("--log", default="research/tune_log.txt")
    ap.add_argument("--params", nargs="*", default=list(GRID))
    a = ap.parse_args()
    snap = Path("research/snapshots"); snap.mkdir(exist_ok=True)
    bot = snap / f"orakel_{int(time.time())}.py"
    shutil.copy("perudo/bots/orakel.py", bot)
    best = {}
    only = None
    log = open(a.log, "a")
    for ps in range(a.passes):
        for name in a.params:
            variants = {}
            for v in GRID[name]:
                cand = dict(best); cand[name] = v
                variants[f"{name}={v}"] = cand
            t = time.time()
            res = run_variants(variants, a.games, a.chunks, seed=100 + ps, only=only, bot=str(bot), ref_params=best)
            print(f"\n== pass {ps} param {name} (base {json.dumps(best)})", file=log)
            sys.stdout = log
            scores = summarize(res, variants)
            sys.stdout = sys.__stdout__
            win = max(scores, key=scores.get)
            best[name] = GRID[name][list(variants).index(win)]
            print(f"-> {name}={best[name]}  ({time.time()-t:.0f}s)", file=log, flush=True)
    print(f"\nFINAL {json.dumps(best)}", file=log, flush=True)

if __name__ == "__main__":
    main()
