"""Behaviour profile: how each bot loses/wins dice (as caller, as bidder, calza)."""
import argparse, json, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perudo.game import PerudoGame
from perudo.models import RevealEvent, BidEvent, DudoEvent, CalzaEvent
from research.arena import build
from research.quick import LINEUPS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineup", default="six")
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    lu = [("bot", "perudo/bots/orakel.py", "OrakelBot", json.loads(a.params), "Orakel")]
    lu += [("zoo", k, f"{k}#{i}") for i, k in enumerate(LINEUPS[a.lineup])]
    bots = [build(s) for s in lu]
    S = defaultdict(lambda: defaultdict(int))
    def listen(e):
        if isinstance(e, BidEvent): S[e.player]["bids"] += 1
        elif isinstance(e, DudoEvent): S[e.player]["dudo"] += 1
        elif isinstance(e, CalzaEvent): S[e.player]["calza"] += 1
        elif isinstance(e, RevealEvent):
            if e.challenge_type == "dudo":
                if e.result == "bidder_correct": S[e.challenger]["dudo_lost"] += 1; S[e.bidder]["bid_survived"] += 1
                else: S[e.bidder]["bid_lost"] += 1; S[e.challenger]["dudo_won"] += 1
            else:
                S[e.challenger]["calza_ok" if e.result == "correct" else "calza_bad"] += 1
    rng = random.Random(5)
    for _ in range(a.games):
        PerudoGame(bots=bots, seed=rng.randrange(2**63), listeners=[listen]).play()
    cols = ["bids", "dudo", "dudo_won", "dudo_lost", "bid_lost", "bid_survived", "calza", "calza_ok", "calza_bad"]
    print(f"{'bot':<14}" + "".join(f"{c:>13}" for c in cols) + f"{'lost/100bids':>13}")
    for b in bots:
        s = S[b.name]
        lost = s["dudo_lost"] + s["bid_lost"] + s["calza_bad"]
        print(f"{b.name:<14}" + "".join(f"{s[c]:>13}" for c in cols) + f"{100*lost/max(1,s['bids']+s['dudo']+s['calza']):>13.1f}")

if __name__ == "__main__":
    main()
