"""
Palifico-specific engine tests. These poke the engine's internal
(private) helpers directly with controlled player state, so outcomes are
fully deterministic and don't depend on dice RNG.
"""

from perudo.game import PerudoGame
from perudo.models import Bid
from perudo.rules import count_matches, legal_bids


class DummyBot:
    def __init__(self, name):
        self.name = name


def make_game(n=2, seed=1):
    return PerudoGame(bots=[DummyBot(f"P{i}") for i in range(n)], seed=seed)


def test_palifico_triggers_on_first_2_to_1_transition():
    game = make_game()
    player = game.players[0]
    player.dice_count = 2
    triggered = game._lose_die(0)
    assert triggered is True
    assert player.dice_count == 1
    assert player.has_had_palifico is True


def test_palifico_triggers_at_most_once_per_player():
    game = make_game()
    player = game.players[0]

    player.dice_count = 2
    assert game._lose_die(0) is True  # first 2 -> 1: triggers

    # Player recovers back to 2 dice (e.g. via a correct Calza elsewhere).
    player.dice_count = 2
    assert game._lose_die(0) is False  # second 2 -> 1: no new Palifico


def test_palifico_does_not_trigger_on_elimination():
    game = make_game()
    player = game.players[0]
    player.dice_count = 1
    triggered = game._lose_die(0)
    assert triggered is False
    assert player.alive is False


def test_palifico_works_with_exactly_two_players():
    game = make_game(n=2)
    assert len(game.players) == 2
    player = game.players[1]
    player.dice_count = 2
    assert game._lose_die(1) is True


def test_paco_is_not_wild_during_palifico():
    assert count_matches([1, 1, 4], face=4, palifico=True) == 1
    assert count_matches([1, 1, 4], face=4, palifico=False) == 3


def test_face_cannot_change_during_a_palifico_round():
    bids = legal_bids(Bid(2, 3), palifico=True, palifico_face=3, total_dice=15)
    assert all(b.face == 3 for b in bids)


def test_opening_bid_in_palifico_round_may_be_paco():
    bids = legal_bids(None, palifico=True, palifico_face=None, total_dice=15)
    assert any(b.face == 1 for b in bids)
