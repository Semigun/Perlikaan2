"""
Opponent zoo for testing: stand-ins for the tournament bots we cannot see.

The organiser's .gitignore hides 110iq, 120iq, 130iq, Autist and Believer,
so these are educated guesses at what such bots might do, plus a spread of
other plausible styles (cautious, aggressive bluffer, Calza lover, ...).
Every bot here is configurable through keyword arguments so the arena can
also generate random "population" lineups.

None of these live in perudo/bots/, so the official loader never sees them.
"""

from __future__ import annotations

import importlib.util
import math
import random
from functools import lru_cache
from pathlib import Path

from perudo.models import PACO, Bid, Calza, Dudo
from perudo.rules import count_matches

ROOT = Path(__file__).resolve().parent.parent


def _load_repo_bot(filename: str, class_name: str):
    path = ROOT / "perudo" / "bots" / filename
    spec = importlib.util.spec_from_file_location(f"zoo_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


HundredIQBot = _load_repo_bot("100iq.py", "HundredIQBot")
MinRaiseBot = _load_repo_bot("minraise.py", "MinRaiseBot")
RandomBot = _load_repo_bot("random_bot.py", "RandomBot")
ExampleBot = _load_repo_bot("example_bot.py", "ExampleBot")


@lru_cache(maxsize=None)
def p_at_least(n: int, k: int, p6: int) -> float:
    """P(Binomial(n, p6/6) >= k)."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    p = p6 / 6.0
    return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(k, n + 1))


@lru_cache(maxsize=None)
def p_exactly(n: int, k: int, p6: int) -> float:
    if k < 0 or k > n:
        return 0.0
    p = p6 / 6.0
    return math.comb(n, k) * p**k * (1 - p) ** (n - k)


def _p6(face, palifico):
    return 1 if (palifico or face == PACO) else 2


class ParamBot:
    """
    A configurable "reasonable" probability bot.

    - dudo_t:     call Dudo when P(current bid true) < dudo_t
    - compare:    if True, call Dudo when P(bid false) > P(best raise true)
    - calza_t:    call Calza when P(exact) > calza_t (0 disables)
    - believe:    number of dice each earlier bidder is assumed to hold of
                  the face they bid (0 = ignore other bids)
    - bluff:      probability of bidding a random face instead of the best
    - open_mode:  "min" (1x face), "expect" (expected count) or "expect-1"
    - raise_mode: "best" (max probability) or "cheapest" (cheapest on best face)
    - noise:      random jitter on thresholds
    """

    def __init__(self, name, dudo_t=0.5, compare=False, calza_t=0.0, believe=0,
                 bluff=0.0, open_mode="expect-1", raise_mode="best", noise=0.0,
                 seed=None):
        self.name = name
        self.dudo_t = dudo_t
        self.compare = compare
        self.calza_t = calza_t
        self.believe = believe
        self.bluff = bluff
        self.open_mode = open_mode
        self.raise_mode = raise_mode
        self.noise = noise
        self.rng = random.Random(seed)

    def _known(self, state, face):
        """Matches I know about: my own dice (+ believed dice of other bidders)."""
        known = count_matches(state.my_dice, face, state.palifico)
        claimed_by = {}
        if self.believe:
            me = state.players[state.my_id].name
            for rec in state.history:
                if isinstance(rec.action, Bid) and rec.player != me:
                    a = rec.action
                    if a.face == face or (not state.palifico and a.face == PACO and face != PACO):
                        claimed_by[rec.player] = True
        return known, len(claimed_by)

    def _p_true(self, state, bid):
        total = sum(p.dice_count for p in state.players)
        unknown = total - state.my_dice_count
        known, believers = self._known(state, bid.face)
        extra = 0
        if believers:
            # each believed bidder is assumed to hold `believe` matches
            extra = believers * self.believe
            unknown = max(0, unknown - extra)
        return p_at_least(unknown, bid.quantity - known - extra, _p6(bid.face, state.palifico))

    def _p_exact(self, state, bid):
        total = sum(p.dice_count for p in state.players)
        unknown = total - state.my_dice_count
        known, _ = self._known(state, bid.face)
        return p_exactly(unknown, bid.quantity - known, _p6(bid.face, state.palifico))

    def _expected(self, state, face):
        total = sum(p.dice_count for p in state.players)
        unknown = total - state.my_dice_count
        known = count_matches(state.my_dice, face, state.palifico)
        return known + unknown * _p6(face, state.palifico) / 6.0

    def play(self, state):
        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        jitter = self.rng.uniform(-self.noise, self.noise) if self.noise else 0.0

        if state.current_bid is not None:
            p_cur = self._p_true(state, state.current_bid)
            if self.calza_t and self._p_exact(state, state.current_bid) > self.calza_t + jitter:
                return Calza()
            if not bids:
                return Dudo()
            if p_cur < self.dudo_t + jitter:
                return Dudo()

        if not bids:
            return Dudo()

        if self.bluff and self.rng.random() < self.bluff:
            faces = sorted({b.face for b in bids})
            face = self.rng.choice(faces)
            return min((b for b in bids if b.face == face), key=lambda b: b.quantity)

        if state.current_bid is None:
            faces = range(1, 7) if state.palifico else range(2, 7)
            best_face = max(faces, key=lambda f: (self._expected(state, f) * (2 if (f == PACO and not state.palifico) else 1), f))
            exp = self._expected(state, best_face)
            if self.open_mode == "min":
                q = 1
            elif self.open_mode == "expect":
                q = int(exp)
            else:
                q = int(exp) - 1
            q = max(1, q)
            cands = [b for b in bids if b.face == best_face and b.quantity == q]
            if cands:
                return cands[0]
            return min((b for b in bids if b.face == best_face), key=lambda b: b.quantity)

        if self.raise_mode == "cheapest":
            faces = range(1, 7)
            best_face = max(faces, key=lambda f: (count_matches(state.my_dice, f, state.palifico), f))
            same = [b for b in bids if b.face == best_face]
            if same:
                return min(same, key=lambda b: b.quantity)
            return min(bids, key=lambda b: (b.quantity, b.face))

        # "best": the cheapest raise per face, pick the most probable one
        cheapest = {}
        for b in bids:
            c = cheapest.get(b.face)
            if c is None or b.quantity < c.quantity:
                cheapest[b.face] = b
        scored = [(self._p_true(state, b), -b.quantity, b.face, b) for b in cheapest.values()]
        scored.sort(key=lambda t: t[:3], reverse=True)
        best_p, _, _, best_bid = scored[0]
        if self.compare and state.current_bid is not None:
            if 1 - self._p_true(state, state.current_bid) > best_p:
                return Dudo()
        return best_bid


class BelieverBot(ParamBot):
    """Trusts other players' bids (assumes bidders hold their face) and
    almost never calls Dudo unless a bid is near-impossible."""

    def __init__(self, name="Believer", **kw):
        kw.setdefault("dudo_t", 0.2)
        kw.setdefault("believe", 1)
        kw.setdefault("open_mode", "expect-1")
        super().__init__(name, **kw)


class AutistBot:
    """Pure expected-value machine: bids exactly the expected count of its
    best face and calls Dudo whenever the bid exceeds the expectation."""

    def __init__(self, name="Autist", margin=0.5):
        self.name = name
        self.margin = margin

    def play(self, state):
        total = sum(p.dice_count for p in state.players)
        unknown = total - state.my_dice_count

        def expected(face):
            p6 = _p6(face, state.palifico)
            return count_matches(state.my_dice, face, state.palifico) + unknown * p6 / 6.0

        bids = [a for a in state.legal_actions if isinstance(a, Bid)]
        if state.current_bid is not None:
            if not bids or state.current_bid.quantity > expected(state.current_bid.face) + self.margin:
                return Dudo()
        if not bids:
            return Dudo()
        # cheapest legal bid whose quantity does not exceed its expectation
        ok = [b for b in bids if b.quantity <= expected(b.face)]
        if ok:
            return max(ok, key=lambda b: (expected(b.face) - b.quantity, -b.quantity, b.face))
        if state.current_bid is None:
            return min(bids, key=lambda b: (b.quantity, -b.face))
        return Dudo()


def make_110(name="110 IQ"):
    # 100IQ + pacos counted as wild when picking a face + sensible opening
    return ParamBot(name, dudo_t=0.5, raise_mode="best", open_mode="expect-1")


def make_120(name="120 IQ"):
    return ParamBot(name, dudo_t=0.45, compare=True, calza_t=0.35, open_mode="expect-1")


def make_130(name="130 IQ"):
    return ParamBot(name, dudo_t=0.42, compare=True, calza_t=0.33, believe=1, open_mode="expect")


def make_cautious(name="Cautious"):
    return ParamBot(name, dudo_t=0.62, open_mode="expect-1")


def make_aggro(name="Bluffer"):
    return ParamBot(name, dudo_t=0.3, bluff=0.35, open_mode="expect", noise=0.1)


def make_calza(name="Calza Fan"):
    return ParamBot(name, dudo_t=0.5, calza_t=0.22, open_mode="expect-1")


def make_100(name="100 IQ"):
    b = HundredIQBot()
    b.name = name
    b.count = 1
    return b


def make_minraise(name="MinRaise"):
    b = MinRaiseBot()
    b.name = name
    return b


def make_random(name="Random"):
    b = RandomBot()
    b.name = name
    return b


def make_example(name="Example"):
    b = ExampleBot()
    b.name = name
    return b


def make_believer(name="Believer"):
    return BelieverBot(name)


def make_autist(name="Autist"):
    return AutistBot(name)


# Strongest heuristic bots found by research/evolve.py at 7-player tables
# (a population of ParamBots evolved against each other).
def make_evo1(name="Evo1"):
    return ParamBot(name, dudo_t=0.183, open_mode="expect", raise_mode="best")


def make_evo2(name="Evo2"):
    return ParamBot(name, dudo_t=0.150, open_mode="min", raise_mode="best")


def make_evo3(name="Evo3"):
    return ParamBot(name, dudo_t=0.239, believe=2, bluff=0.127, open_mode="min",
                    raise_mode="best", noise=0.05, seed=7)


FACTORIES = {
    "evo1": make_evo1,
    "evo2": make_evo2,
    "evo3": make_evo3,
    "100": make_100,
    "110": make_110,
    "120": make_120,
    "130": make_130,
    "believer": make_believer,
    "autist": make_autist,
    "cautious": make_cautious,
    "bluffer": make_aggro,
    "calza": make_calza,
    "minraise": make_minraise,
    "random": make_random,
    "example": make_example,
}


def random_parambot(rng: random.Random, name: str) -> ParamBot:
    return ParamBot(
        name,
        dudo_t=rng.uniform(0.25, 0.65),
        compare=rng.random() < 0.5,
        calza_t=rng.choice([0.0, 0.0, rng.uniform(0.2, 0.45)]),
        believe=rng.choice([0, 0, 1, 2]),
        bluff=rng.choice([0.0, 0.0, rng.uniform(0.05, 0.4)]),
        open_mode=rng.choice(["min", "expect", "expect-1"]),
        raise_mode=rng.choice(["best", "best", "cheapest"]),
        noise=rng.choice([0.0, 0.05, 0.15]),
        seed=rng.randrange(1 << 30),
    )
