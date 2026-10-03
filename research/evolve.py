"""
Evolve strong heuristic (ParamBot) opponents at 7-player tables, to get
a tougher and more realistic test field than hand-written zoo bots.
Prints the best parameter sets found.
"""
import json, random, sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perudo.game import PerudoGame
from research.zoo import ParamBot

KEYS = ["dudo_t", "compare", "calza_t", "believe", "bluff", "open_mode", "raise_mode", "noise"]

def rand_params(rng):
    return dict(dudo_t=rng.uniform(0.1, 0.6), compare=rng.random() < 0.5,
                calza_t=rng.choice([0.0, rng.uniform(0.2, 0.5)]), believe=rng.choice([0, 1, 2]),
                bluff=rng.choice([0.0, rng.uniform(0.0, 0.3)]), open_mode=rng.choice(["min", "expect", "expect-1"]),
                raise_mode=rng.choice(["best", "cheapest"]), noise=rng.choice([0.0, 0.05, 0.1]))

def mutate(p, rng):
    q = dict(p)
    k = rng.choice(KEYS)
    q.update({k: rand_params(rng)[k]})
    if rng.random() < 0.5:
        q["dudo_t"] = min(0.7, max(0.05, q["dudo_t"] + rng.gauss(0, 0.05)))
    return q

def table(args):
    plist, games, seed = args
    bots = [ParamBot(f"b{i}", seed=seed + i, **p) for i, p in enumerate(plist)]
    rng = random.Random(seed)
    wins = [0] * len(bots)
    for _ in range(games):
        w = PerudoGame(bots=bots, seed=rng.randrange(2**63)).play().winner
        wins[int(w[1:])] += 1
    return wins

def main():
    rng = random.Random(42)
    pop = [rand_params(rng) for _ in range(42)]
    pop[0] = dict(dudo_t=0.2, compare=False, calza_t=0.0, believe=1, bluff=0.0, open_mode="expect-1", raise_mode="best", noise=0.0)
    with ProcessPoolExecutor() as ex:
        for gen in range(6):
            score = [0.0] * len(pop); seen = [0] * len(pop)
            jobs, idxs = [], []
            for t in range(36):
                ids = rng.sample(range(len(pop)), 7)
                jobs.append(([pop[i] for i in ids], 150, rng.randrange(1 << 30))); idxs.append(ids)
            for ids, wins in zip(idxs, ex.map(table, jobs)):
                for i, w in zip(ids, wins):
                    score[i] += w / 150; seen[i] += 1
            avg = [score[i] / max(1, seen[i]) * 7 for i in range(len(pop))]
            order = sorted(range(len(pop)), key=lambda i: -avg[i])
            print(f"gen {gen}: best {avg[order[0]]:.2f} {json.dumps(pop[order[0]])}", flush=True)
            elite = [pop[i] for i in order[:14]]
            pop = elite + [mutate(rng.choice(elite), rng) for _ in range(28)]
    print("TOP5")
    for p in elite[:5]:
        print(json.dumps(p))

if __name__ == "__main__":
    main()
