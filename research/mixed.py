"""Evaluate a set of Orakel variants against each other plus zoo bots."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.arena import evaluate, report

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default='{"A": {}}', help="json name->params")
    ap.add_argument("--zoo", nargs="*", default=["believer", "130", "110", "100"])
    ap.add_argument("--games", type=int, default=250)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    variants = json.loads(a.variants)
    lu = [("bot", "perudo/bots/orakel.py", "OrakelBot", p, n) for n, p in variants.items()]
    lu += [("zoo", k, f"{k}#{i}") for i, k in enumerate(a.zoo)]
    t = time.time()
    report(evaluate([lu] * a.chunks, a.games, a.seed))
    print(f"{time.time()-t:.1f}s  fair share {100/len(lu):.1f}%")

if __name__ == "__main__":
    main()
