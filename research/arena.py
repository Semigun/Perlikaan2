"""
Parallel arena: play many games with a given lineup and report win rates.

A lineup is a list of picklable specs so it can be rebuilt inside worker
processes:

    ("zoo", "<factory key>", "<display name>")
    ("bot", "<path/to/file.py>", "<ClassName>", {kwargs}, "<display name>")

Each worker process plays `games` games as one continuous "tournament"
(the same bot instances are reused across games, exactly like
run_simulation.py), so learning bots get to learn within a chunk.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from perudo.game import PerudoGame  # noqa: E402
from research import zoo  # noqa: E402

_MODULE_CACHE = {}


def load_class(path: str, class_name: str):
    key = (path, class_name)
    if key not in _MODULE_CACHE:
        full = Path(path)
        if not full.is_absolute():
            full = ROOT / full
        spec = importlib.util.spec_from_file_location(f"arena_{full.stem}_{abs(hash(str(full)))}", full)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _MODULE_CACHE[key] = getattr(module, class_name)
    return _MODULE_CACHE[key]


def build(spec):
    kind = spec[0]
    if kind == "zoo":
        _, key, name = spec
        return zoo.FACTORIES[key](name)
    if kind == "pop":
        _, seed, name = spec
        return zoo.random_parambot(random.Random(seed), name)
    if kind == "bot":
        _, path, cls_name, kwargs, name = spec
        bot = load_class(path, cls_name)(**kwargs)
        bot.name = name
        return bot
    raise ValueError(spec)


def play_chunk(args):
    lineup, games, seed = args
    bots = [build(s) for s in lineup]
    seed_rng = random.Random(seed)
    wins = defaultdict(int)
    pos = defaultdict(int)
    for _ in range(games):
        result = PerudoGame(bots=bots, seed=seed_rng.randrange(2**63)).play()
        wins[result.winner] += 1
        for i, name in enumerate(result.standings, start=1):
            pos[name] += i
    return dict(wins), dict(pos), games


def evaluate(lineups, games, seed=0, workers=None):
    """lineups: list of lineups (each a list of specs). Every lineup is
    played for `games` games as one tournament. Returns aggregated stats."""
    jobs = [(lu, games, seed * 1_000_003 + i) for i, lu in enumerate(lineups)]
    workers = workers or os.cpu_count()
    wins = defaultdict(int)
    pos = defaultdict(int)
    played = defaultdict(int)
    if workers == 1:
        results = map(play_chunk, jobs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        results = ex.map(play_chunk, jobs)
    for (lu, _, _), (w, p, g) in zip(jobs, results):
        for spec in lu:
            name = spec[-1]
            played[name] += g
            wins[name] += w.get(name, 0)
            pos[name] += p.get(name, 0)
    if workers != 1:
        ex.shutdown()
    return {name: (wins[name], played[name], pos[name] / played[name]) for name in played}


def report(stats, sizes=None):
    rows = sorted(stats.items(), key=lambda kv: -kv[1][0] / kv[1][1])
    print(f"{'bot':<26}{'wins':>8}{'games':>8}{'win%':>8}{'±2se':>7}{'avgpos':>8}")
    for name, (w, g, ap) in rows:
        p = w / g
        se = math.sqrt(max(p * (1 - p), 1e-9) / g)
        print(f"{name:<26}{w:>8}{g:>8}{100*p:>7.1f}%{200*se:>6.1f}{ap:>8.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=500)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--zoo", nargs="+", default=["100", "110", "120", "130", "believer", "autist"])
    args = ap.parse_args()
    lineup = [("zoo", k, k) for k in args.zoo]
    t = time.time()
    stats = evaluate([lineup] * args.chunks, args.games, args.seed)
    report(stats)
    print(f"{time.time() - t:.1f}s")


if __name__ == "__main__":
    main()
