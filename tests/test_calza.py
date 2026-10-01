"""
Calza-specific engine tests, using the engine's internal resolution
helpers directly with hand-crafted dice so outcomes are deterministic.
"""

from perudo.game import PerudoGame
from perudo.models import Bid, Calza
from perudo.rules import legal_actions


class DummyBot:
    def __init__(self, name):
        self.name = name


def make_game(n=2, seed=1):
    return PerudoGame(bots=[DummyBot(f"P{i}") for i in range(n)], seed=seed)


def test_correct_calza_returns_a_die_before_first_elimination():
    game = make_game()
    caller_idx, bidder_idx = 0, 1
    caller = game.players[caller_idx]
    bidder = game.players[bidder_idx]
    caller.dice_count, caller.dice = 3, [4, 4, 1]  # two 4s + one Paco = 3
    bidder.dice_count, bidder.dice = 2, [6, 6]

    bid = Bid(3, 4)
    next_starter, palifico_triggered = game._resolve_calza(caller_idx, bidder_idx, bid, palifico=False)

    assert caller.dice_count == 4
    assert next_starter == caller_idx
    assert palifico_triggered is False


def test_incorrect_calza_loses_a_die():
    game = make_game()
    caller_idx, bidder_idx = 0, 1
    caller = game.players[caller_idx]
    bidder = game.players[bidder_idx]
    caller.dice_count, caller.dice = 3, [2, 3, 5]
    bidder.dice_count, bidder.dice = 2, [6, 6]

    bid = Bid(3, 4)  # actual = 0, not 3
    next_starter, _ = game._resolve_calza(caller_idx, bidder_idx, bid, palifico=False)

    assert caller.dice_count == 2
    assert next_starter == caller_idx


def test_calza_off_by_one_either_way_is_wrong():
    game = make_game()
    caller_idx, bidder_idx = 0, 1
    caller = game.players[caller_idx]
    bidder = game.players[bidder_idx]

    # actual = 2, bid = 3 (one too many)
    caller.dice_count, caller.dice = 2, [4, 1]
    bidder.dice_count, bidder.dice = 2, [6, 6]
    game._resolve_calza(caller_idx, bidder_idx, Bid(3, 4), palifico=False)
    assert caller.dice_count == 1  # lost a die

    # actual = 2, bid = 1 (one too few)
    caller.dice_count, caller.dice = 2, [4, 1]
    game._resolve_calza(caller_idx, bidder_idx, Bid(1, 4), palifico=False)
    assert caller.dice_count == 1  # lost a die again


def test_calza_never_exceeds_five_dice():
    game = make_game()
    caller_idx, bidder_idx = 0, 1
    caller = game.players[caller_idx]
    bidder = game.players[bidder_idx]
    caller.dice_count, caller.dice = 5, [4, 4, 1, 2, 2]
    bidder.dice_count, bidder.dice = 2, [6, 6]

    game._resolve_calza(caller_idx, bidder_idx, Bid(3, 4), palifico=False)
    assert caller.dice_count == 5  # capped


def test_recovery_disabled_after_first_elimination_in_the_game():
    game = make_game(n=3)
    game.dice_recovery_enabled = False  # someone was already eliminated earlier
    caller_idx, bidder_idx = 0, 1
    caller = game.players[caller_idx]
    bidder = game.players[bidder_idx]
    caller.dice_count, caller.dice = 3, [4, 4, 1]
    bidder.dice_count, bidder.dice = 2, [6, 6]

    game._resolve_calza(caller_idx, bidder_idx, Bid(3, 4), palifico=False)
    assert caller.dice_count == 3  # correct, but no die returned


def test_recovery_disabled_incorrect_calza_still_loses_die():
    game = make_game(n=3)
    game.dice_recovery_enabled = False
    caller_idx, bidder_idx = 0, 1
    caller = game.players[caller_idx]
    bidder = game.players[bidder_idx]
    caller.dice_count, caller.dice = 3, [2, 3, 5]
    bidder.dice_count, bidder.dice = 2, [6, 6]

    game._resolve_calza(caller_idx, bidder_idx, Bid(3, 4), palifico=False)
    assert caller.dice_count == 2


def test_incorrect_calza_can_eliminate_the_caller():
    game = make_game(n=3)
    caller_idx, bidder_idx = 0, 1
    caller = game.players[caller_idx]
    bidder = game.players[bidder_idx]
    caller.dice_count, caller.dice = 1, [2]
    bidder.dice_count, bidder.dice = 2, [6, 6]

    next_starter, _ = game._resolve_calza(caller_idx, bidder_idx, Bid(3, 4), palifico=False)

    assert caller.alive is False
    assert game.dice_recovery_enabled is False
    assert next_starter != caller_idx
    assert game.players[next_starter].alive


def test_palifico_edge_case_recover_to_two_then_lose_again_no_new_palifico():
    game = make_game()
    caller_idx, bidder_idx = 0, 1
    caller = game.players[caller_idx]
    bidder = game.players[bidder_idx]

    # Caller has 1 die, correct Calza brings them to 2 -- nothing special.
    caller.dice_count, caller.dice = 1, [4]
    bidder.dice_count, bidder.dice = 2, [1, 1]  # two Pacos match face 4
    _, triggered = game._resolve_calza(caller_idx, bidder_idx, Bid(3, 4), palifico=False)
    assert caller.dice_count == 2
    assert triggered is False
    assert caller.has_had_palifico is False

    # Now they lose a die again, going 2 -> 1: this is their FIRST ever
    # Palifico trigger (the earlier 1 -> 2 gain was not a loss).
    triggered_again = game._lose_die(caller_idx)
    assert triggered_again is True
    assert caller.has_had_palifico is True


def test_calza_is_only_ever_offered_to_the_player_whose_turn_it_is():
    # legal_actions only ever includes Calza once a bid exists, and
    # PerudoGame only ever calls play() (and therefore only ever computes
    # legal_actions) for the player whose turn it currently is -- so there
    # is structurally no way to call Calza off-turn.
    assert Calza() not in legal_actions(None, palifico=False, palifico_face=None, total_dice=10)
    assert Calza() in legal_actions(Bid(1, 2), palifico=False, palifico_face=None, total_dice=10)
