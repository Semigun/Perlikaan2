"""Tests for Bas de Batser: static (non-learning) bot that always opens 1x 2."""

import importlib.util
import random
from pathlib import Path

from perudo.game import PerudoGame
from perudo.models import Bid

BOTS_DIR = Path(__file__).resolve().parent.parent / "perudo" / "bots"


def _load(filename, class_name):
    spec = importlib.util.spec_from_file_location(f"test_{filename[:-3]}", BOTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


Bas = _load("bas_de_batser.py", "BasDeBatserBot")
Orakel = _load("orakel.py", "OrakelBot")
HundredIQ = _load("100iq.py", "HundredIQBot")


def _lineup():
    opp = HundredIQ()
    opp.name = "100"
    return [Bas(strict=True), Orakel(strict=True), opp]


def test_name():
    assert Bas().name == "Bas de Batser"


def test_always_opens_with_one_two():
    bots = _lineup()
    bas = bots[0]
    openings = []
    real = bas.play

    def spy(state):
        action = real(state)
        if state.current_bid is None:
            openings.append(action)
        return action

    bas.play = spy
    rng = random.Random(1)
    for _ in range(15):
        PerudoGame(bots=bots, seed=rng.randrange(2**32)).play()
    assert openings and all(a == Bid(1, 2) for a in openings)


def test_does_not_learn():
    bots = _lineup()
    for seed in range(10):
        PerudoGame(bots=bots, seed=seed).play()
    bas = bots[0]
    assert bas.stats == {} and bas.pop.n_calls == 0 and bas.rb == {}
