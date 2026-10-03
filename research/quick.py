"""Quick evaluation of Orakel (with optional param overrides) vs zoo lineups."""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from research.arena import evaluate, report

LINEUPS = {
    "six": ["100", "110", "120", "130", "believer"],
    "ten": ["100", "110", "120", "130", "believer", "autist", "cautious", "bluffer", "calza"],
    "three": ["believer", "130"],
    "duel100": ["100"],
    "duelbel": ["believer"],
    "iq": ["100", "100", "100"],
    "evo7": ["evo1", "evo2", "evo3", "evo1", "believer", "evo2"],
    "friends7": ["100", "110", "120", "130", "autist", "believer"],
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineup", default="six")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--params", default="{}")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    params = json.loads(a.params)
    keys = LINEUPS[a.lineup]
    lu = [("bot", "perudo/bots/orakel.py", "OrakelBot", params, "Orakel")]
    lu += [("zoo", k, f"{k}#{i}") for i, k in enumerate(keys)]
    t = time.time()
    st = evaluate([lu] * a.chunks, a.games, a.seed)
    report(st)
    print(f"{time.time()-t:.1f}s  baseline share = {100/len(lu):.1f}%")

if __name__ == "__main__":
    main()
