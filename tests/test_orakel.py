"""
Tests for the Orakel bot: it must always return a legal action, never
crash, make the obvious calls, and keep what it learned across games.
"""

import importlib.util
import random
from pathlib import Path

from perudo.game import PerudoGame
from perudo.models import (
    ActionRecord,
    Bid,
    Calza,
    Dudo,
    Observation,
    PublicPlayer,
)
from perudo.rules import legal_actions

BOTS_DIR = Path(__file__).resolve().parent.parent / "perudo" / "bots"


def _load(filename, class_name):
    spec = importlib.util.spec_from_file_location(f"test_{filename[:-3]}", BOTS_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, class_name)


OrakelBot = _load("orakel.py", "OrakelBot")
HundredIQBot = _load("100iq.py", "HundredIQBot")
RandomBot = _load("random_bot.py", "RandomBot")
MinRaiseBot = _load("minraise.py", "MinRaiseBot")


def _named(bot, name):
    bot.name = name
    return bot


def _obs(my_dice, others, current_bid=None, palifico=False, history=(), all_alive=True):
    players = [PublicPlayer(id=0, name="Orakel", dice_count=len(my_dice), alive=True)]
    for i, (name, n) in enumerate(others, start=1):
        players.append(PublicPlayer(id=i, name=name, dice_count=n, alive=n > 0))
    if not all_alive:
        players.append(PublicPlayer(id=len(players), name="Gone", dice_count=0, alive=False))
    total = sum(p.dice_count for p in players)
    face = current_bid.face if (palifico and current_bid) else None
    return Observation(
        my_id=0,
        my_dice=tuple(sorted(my_dice)),
        my_dice_count=len(my_dice),
        players=tuple(players),
        current_bid=current_bid,
        palifico=palifico,
        round_number=1,
        history=tuple(history),
        legal_actions=tuple(legal_actions(current_bid, palifico, face, total)),
    )


def test_full_games_with_mixed_opponents_are_always_legal():
    # The engine raises IllegalActionError on any illegal action, so simply
    # finishing many games proves legality across table sizes. strict=True
    # makes internal errors surface instead of using the safety fallback.
    rng = random.Random(4)
    for n_opp in (1, 2, 4, 6, 11):
        bots = [OrakelBot(strict=True)]
        for i in range(n_opp):
            cls = rng.choice([HundredIQBot, RandomBot, MinRaiseBot])
            bots.append(_named(cls(), f"opp{i}"))
        for _ in range(8):
            PerudoGame(bots=bots, seed=rng.randrange(2**32)).play()


def test_self_play_runs():
    bots = [OrakelBot(strict=True), _named(OrakelBot(strict=True, explore_eps=0.002), "Orakel 2"),
            _named(OrakelBot(strict=True, accept_pow=0.5), "Orakel 3")]
    for seed in range(5):
        PerudoGame(bots=bots, seed=seed).play()


def test_calls_dudo_on_impossible_bid():
    bot = OrakelBot()
    obs = _obs((2, 3, 4), [("A", 2)], current_bid=Bid(5, 6),
               history=[ActionRecord("A", Bid(5, 6))])
    assert bot.play(obs) == Dudo()


def test_never_calls_dudo_when_own_dice_prove_the_bid():
    bot = OrakelBot()
    obs = _obs((1, 5, 5, 5, 5), [("A", 5), ("B", 5)], current_bid=Bid(5, 5),
               history=[ActionRecord("A", Bid(5, 5))])
    assert bot.play(obs) not in (Dudo(),)


def test_opening_bid_is_legal_and_not_paco():
    bot = OrakelBot()
    obs = _obs((2, 2, 3, 6, 6), [("A", 5), ("B", 5), ("C", 5)])
    action = bot.play(obs)
    assert isinstance(action, Bid) and action.face != 1
    assert action in obs.legal_actions


def test_maxed_out_palifico_round_only_dudo_or_calza():
    bot = OrakelBot()
    obs = _obs((4,), [("A", 1)], current_bid=Bid(2, 4), palifico=True,
               history=[ActionRecord("A", Bid(2, 4))])
    action = bot.play(obs)
    assert action in (Dudo(), Calza())


def test_handles_unknown_names_and_weird_history_without_crashing():
    bot = OrakelBot()
    obs = _obs((3, 3), [("A", 2)], current_bid=Bid(1, 3),
               history=[ActionRecord("Stranger", Bid(1, 3))])
    assert bot.play(obs) in obs.legal_actions


def test_learning_persists_across_games():
    me = OrakelBot(strict=True)
    opp = _named(HundredIQBot(), "Honest")
    for seed in range(30):
        PerudoGame(bots=[me, opp, _named(HundredIQBot(), "Other")], seed=seed).play()
    stats = me.stats["Honest"]
    assert stats.rounds > 50
    assert stats.n_calls > 50
    # 100 IQ always bids the face it holds most, so Orakel must have
    # learned that "holds none of the bid face" is much rarer than chance.
    key = max(stats.bid, key=lambda k: sum(stats.bid[k]))
    lr = me._likelihood_ratio("Honest", *key)
    assert lr[0] < 0.5
