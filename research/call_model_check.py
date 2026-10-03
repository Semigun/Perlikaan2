"""
Which feature best predicts when an opponent calls Dudo?

  naive : P(bid true) from the caller's own dice + binomial for the rest
  bayes : same, but the rest of the table is read through the bidders'
          learned honesty (what a Bayesian opponent would compute)

Logs every opponent decision in strong 7-player games, then compares the
held-out log-loss of binned models on each feature (and both together).
"""
import math, random, sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perudo.game import PerudoGame
from perudo.models import Bid, Dudo, Calza
from research.arena import build, load_class

OK = load_class("perudo/bots/orakel.py", "OrakelBot")
G = OK._play.__globals__

def collect(lineup, games, seed):
    bots = [build(s) for s in lineup]
    trained = bots[0]
    rows = []
    pending = {}
    def before_turn(obs):
        if obs.current_bid is None:
            pending.clear(); return
        name = obs.players[obs.my_id].name
        bid = obs.current_bid
        pal = obs.palifico
        p6 = G["_p6"](bid.face, pal)
        total = sum(p.dice_count for p in obs.players)
        k = sum(1 for d in obs.my_dice if d == bid.face or (not pal and bid.face != 1 and d == 1))
        x_naive = G["_binom_tail"](total - obs.my_dice_count, bid.quantity - k, p6)
        ob = OK(); ob.name = name; ob.stats = trained.stats; ob.pop = trained.pop
        ob._lr_cache = {}; ob._call_cache = {}
        ob._last_pt = None
        try:
            ob._play(obs)
        except Exception:
            pass
        x_bayes = ob._last_pt if ob._last_pt is not None else x_naive
        last = obs.history[-1].action
        pending["row"] = (name.split("#")[0].rstrip("0123456789"), x_naive, x_bayes)
    def listen(e):
        from perudo.models import BidEvent, DudoEvent, CalzaEvent
        if "row" in pending and isinstance(e, (BidEvent, DudoEvent, CalzaEvent)):
            kind, xn, xb = pending.pop("row")
            rows.append((kind, xn, xb, isinstance(e, DudoEvent)))
    rng = random.Random(seed)
    for _ in range(games):
        PerudoGame(bots=bots, seed=rng.randrange(2**63), before_turn=before_turn, listeners=[listen]).play()
    return rows

def binmodel(train, test, feat, nb=10):
    cnt = defaultdict(lambda: [1, 2])  # laplace
    def key(r):
        if feat == "both":
            return (min(nb-1, int(r[1]*nb)), min(nb-1, int(r[2]*nb)))
        return min(nb-1, int(r[1 if feat == "naive" else 2]*nb))
    for r in train:
        c = cnt[key(r)]; c[0] += r[3]; c[1] += 1
    ll = 0.0
    for r in test:
        c = cnt[key(r)]; p = c[0]/c[1]
        ll -= math.log(p if r[3] else 1-p)
    return ll/len(test)

def main():
    BOT = "perudo/bots/orakel.py"
    lu = [("bot", BOT, "OrakelBot", {}, "Orakel")] + [("bot", BOT, "OrakelBot", {}, f"Ref{i}") for i in range(2)] \
        + [("zoo", k, f"{k}#{i}") for i, k in enumerate(["evo1", "evo2", "evo3", "believer"])]
    rows = collect(lu, 120, 3)
    by = defaultdict(list)
    for r in rows:
        by[r[0]].append(r)
    for kind, rs in sorted(by.items()):
        random.Random(0).shuffle(rs)
        cut = len(rs)*2//3
        tr, te = rs[:cut], rs[cut:]
        base = sum(r[3] for r in tr)/len(tr)
        ll0 = -sum(math.log(base if r[3] else 1-base) for r in te)/len(te)
        print(f"{kind:<10} n={len(rs):5d} dudo-rate={base:.3f}  logloss: const {ll0:.4f}  naive {binmodel(tr,te,'naive'):.4f}  bayes {binmodel(tr,te,'bayes'):.4f}  both {binmodel(tr,te,'both',6):.4f}")

if __name__ == "__main__":
    main()
