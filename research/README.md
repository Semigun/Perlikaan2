# Research tooling for the Orakel bot

Everything in this folder is *offline* tooling used to design and tune
`perudo/bots/orakel.py`. None of it is needed to play: the bot file is
self-contained (standard library + `perudo.models` only).

| file | what it does |
|---|---|
| `zoo.py` | Opponent zoo: the repo's bots plus stand-ins for the hidden 110/120/130 IQ, Autist and Believer bots, a configurable `ParamBot`, and the evolved `Evo1-3` bots |
| `evolve.py` | Evolves strong heuristic `ParamBot`s against each other at 7-player tables (produced Evo1-3) |
| `arena.py` | Parallel game runner. Bot instances are reused across games, like `run_simulation.py`, so learning bots learn |
| `bench.py` | Fixed-seed benchmark suite of 7-player lineups; score = mean(win rate × 7), so 1.0 = fair share |
| `h2h.py`, `h2h_batch.sh` | Candidate vs incumbent in the same 7-player tables (most sensitive test) |
| `tune.py` | Coordinate-descent tuner over `bench.py` |
| `quick.py`, `mixed.py`, `curve.py`, `profile.py`, `inspect_decisions.py` | Quick checks: win rates, learning curve, how dice are lost, decision dumps |
| `call_model_check.py` | Which feature best predicts an opponent's Dudo calls (held-out log-loss) |
| `value_check.py` | Empirical P(win) by dice situation |
| `logs/` | Raw outputs of the parameter head-to-heads and (partial) tuner runs |

Examples:

```bash
python research/quick.py --lineup six --games 300           # Orakel vs zoo
python research/bench.py --variants '{"base": {}, "g2": {"gamma": 2.0}}'
python research/h2h.py --pa '{"gamma": 2.0}'                 # A=candidate, B=default
python research/curve.py --lineup six                        # learning curve
```

## Key findings

1. **Calling Dudo is expensive at a big table.** A Dudo always costs
   someone a die: you or the bidder. Raising passes the risk to the next
   player. Bots that rarely call ("Believer") crush bots that call at
   P(true) < 50% (100 IQ). Evolution converged on calling only below
   P(true) ≈ 0.15–0.18 at 7 players.
2. **Continuation risk depends heavily on table size.** If the next
   player doesn't challenge my bid, the chance I still lose a die this
   round is under 1% at 6+ players but 20–45% in a 1v1. Orakel learns
   this table online from its own bids.
3. **Learning is the main edge.** Turning learning off drops Orakel from
   ~45% to ~25% wins at a 6-player zoo table. Every round ends with a
   reveal of *all* dice, so honesty and calling habits are observable
   for every bid. Most of the gain arrives within the first ~10 games.
4. **Bayesian opponents are predicted better by a Bayesian feature.** For
   Orakel-like and believer-like bots, P(call) is predicted much better
   by "P(true) given the bids so far" than by the naive binomial (log-loss
   0.225 vs 0.265). For naive bots it's the reverse. Orakel learns per
   opponent which model fits, via online Bayesian model averaging.
5. **Calza pays when recovery is still on.** Taken only when its EV beats
   the alternatives, Orakel's Calzas succeed ~60–70% of the time.
6. **What did not help (head-to-head):** random deception among
   near-equal bids (`explore_eps`), "didn't call" evidence
   (`accept_pow`), fewer candidate quantities, and less smoothing of the
   per-opponent honesty model.

## Probability facts used

* Each unknown die matches a normal face with p = 1/3 (face or wild
  Paco), and a Paco (or any face in Palifico) with p = 1/6. So q Pacos is
  about as strong as 2q of a normal face, which is why the switch rules
  halve or double.
* With 30 unknown dice the single most likely exact count of a face has
  only ~15% probability (U=10: 26%, U=4: 40%). That's why Calza is mostly
  an endgame / small-table weapon.
* P(at least q of a face) among 30 dice, no private information:
  q=10 → 57%, q=11 → 42%, q=12 → 28%, q=13 → 17%.
