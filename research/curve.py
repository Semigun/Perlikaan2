"""Learning curve: Orakel's win rate as a function of game index in a tournament."""
import argparse, json, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perudo.game import PerudoGame
from research.arena import build
from research.quick import LINEUPS

BUCKETS = [(0, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 400)]

def run(args):
    lineup, games, seed = args
    bots = [build(s) for s in lineup]
    rng = random.Random(seed)
    wins = []
    for g in range(games):
        r = PerudoGame(bots=bots, seed=rng.randrange(2**63)).play()
        wins.append(r.winner == "Orakel")
    return wins

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineup", default="six")
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    keys = LINEUPS[a.lineup]
    lu = [("bot", "perudo/bots/orakel.py", "OrakelBot", json.loads(a.params), "Orakel")]
    lu += [("zoo", k, f"{k}#{i}") for i, k in enumerate(keys)]
    games = BUCKETS[-1][1]
    with ProcessPoolExecutor() as ex:
        res = list(ex.map(run, [(lu, games, 1000 + i) for i in range(a.reps)]))
    for lo, hi in BUCKETS:
        w = sum(sum(r[lo:hi]) for r in res); n = a.reps * (hi - lo)
        print(f"games {lo:>3}-{hi:<3}: {100*w/n:5.1f}%  (n={n})")

if __name__ == "__main__":
    main()
