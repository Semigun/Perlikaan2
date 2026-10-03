"""
Empirical value of dice at a 7-player table: P(Orakel wins | its dice,
total dice on the table, players alive), measured at every round start,
compared with the bot's value model W = d^g / sum(d^g).
"""
import argparse
import random
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perudo.game import PerudoGame  # noqa: E402
from research.arena import build  # noqa: E402

LINEUP = [("bot", "perudo/bots/orakel.py", "OrakelBot", {}, "Orakel")] + [
    ("zoo", k, f"{k}#{i}") for i, k in enumerate(["100", "110", "120", "130", "autist", "believer"])
]


def run(seed):
    bots = [build(s) for s in LINEUP]
    rng = random.Random(seed)
    rows = []
    for _ in range(150):
        snaps = []

        def before_turn(obs, snaps=snaps):
            if not obs.history:  # first turn of a round
                alive = [p for p in obs.players if p.alive]
                me = next((p for p in alive if p.name == "Orakel"), None)
                if me is not None:
                    others = sorted((p.dice_count for p in alive if p.name != "Orakel"), reverse=True)
                    snaps.append((me.dice_count, tuple(others)))

        res = PerudoGame(bots=bots, seed=rng.randrange(2**63), before_turn=before_turn).play()
        won = res.winner == "Orakel"
        rows.extend((s, won) for s in set(snaps))
    return rows


def model(me, others, g):
    return me ** g / (me ** g + sum(o ** g for o in others))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=16)
    a = ap.parse_args()
    with ProcessPoolExecutor() as ex:
        rows = [r for chunk in ex.map(run, range(a.reps)) for r in chunk]
    by = defaultdict(lambda: [0, 0])
    for (me, others), won in rows:
        key = (len(others) + 1, me, sum(others))
        by[key][0] += won
        by[key][1] += 1
    print(f"{'alive':>5}{'mine':>5}{'others':>7}{'n':>6}{'P(win)':>8}{'W g=1':>7}{'W g=1.5':>8}{'W g=2':>7}")
    # compare with the model using the average shape for that key
    shapes = defaultdict(list)
    for (me, others), won in rows:
        shapes[(len(others) + 1, me, sum(others))].append(others)
    for key in sorted(by):
        w, n = by[key]
        if n < 40:
            continue
        alive, me, tot = key
        sh = shapes[key]
        ms = [sum(model(me, o, g) for o in sh) / len(sh) for g in (1.0, 1.5, 2.0)]
        print(f"{alive:>5}{me:>5}{tot:>7}{n:>6}{w / n:>8.3f}" + "".join(f"{m:>8.3f}" for m in ms))


if __name__ == "__main__":
    main()
