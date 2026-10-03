"""Print Orakel's top-scored options at sampled decision points (debugging aid)."""
import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perudo.game import PerudoGame  # noqa: E402
from research.arena import build  # noqa: E402
from research.quick import LINEUPS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineup", default="six")
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--show", type=int, default=12)
    ap.add_argument("--params", default="{}")
    a = ap.parse_args()
    lu = [("bot", "perudo/bots/orakel.py", "OrakelBot", json.loads(a.params), "Orakel")]
    lu += [("zoo", k, f"{k}#{i}") for i, k in enumerate(LINEUPS[a.lineup])]
    bots = [build(s) for s in lu]
    orakel = bots[0]
    rng = random.Random(9)
    for _ in range(a.warmup):
        PerudoGame(bots=bots, seed=rng.randrange(2**63)).play()

    shown = [0]
    orig = orakel._play

    def spy(state):
        captured = {}
        real_sort = list.sort

        action = orig(state)
        if shown[0] < a.show and rng.random() < 0.15:
            shown[0] += 1
            opts = orakel._last_options
            dice = {p.name: p.dice_count for p in state.players if p.alive}
            print(f"\nround {state.round_number} pal={state.palifico} my dice {state.my_dice} table {dice}")
            print("  history:", ", ".join(f"{r.player}:{r.action}" for r in state.history))
            for ev, _, act in opts[:6]:
                print(f"    {ev:+.4f}  {act}")
            print("  ->", action)
        return action

    orakel._play = spy
    while shown[0] < a.show:
        PerudoGame(bots=bots, seed=rng.randrange(2**63)).play()


if __name__ == "__main__":
    main()
