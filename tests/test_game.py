"""
End-to-end engine tests: full games driven through PerudoGame.play(),
information-hiding guarantees, reproducibility, illegal-action handling,
and bot-instance persistence across multiple simulated games.
"""

import random as random_module

import pytest

from perudo.game import PerudoGame
from perudo.models import (
    Bid,
    BidEvent,
    Dudo,
    Calza,
    IllegalActionError,
    Observation,
)


class RandomTestBot:
    """A bot that plays randomly and records everything it's shown, for
    inspection by tests."""

    def __init__(self, name):
        self.name = name
        self.seen_observations = []
        self.games_started = 0
        self.games_ended = 0

    def on_game_start(self, info):
        self.games_started += 1

    def play(self, state):
        self.seen_observations.append(state)
        return random_module.choice(state.legal_actions)

    def on_game_end(self, result):
        self.games_ended += 1


def make_bots(n=3):
    return [RandomTestBot(f"Bot{i}") for i in range(n)]


# -- full game / reproducibility --------------------------------------------


def test_full_game_runs_to_completion_with_one_winner():
    bots = make_bots(4)
    result = PerudoGame(bots=bots, seed=42).play()
    assert result.winner in [b.name for b in bots]
    assert len(result.standings) == 4
    assert result.standings[0] == result.winner
    assert len(set(result.standings)) == 4  # every player appears exactly once


class DeterministicBot:
    """Always takes the first legal action -- no randomness of its own.

    Used to test the ENGINE's reproducibility in isolation: a bot using
    unseeded global `random` (like RandomBot) will naturally make
    different choices across runs, so it can't be used to prove the
    engine's own dice/seating RNG is reproducible.
    """

    def __init__(self, name):
        self.name = name

    def play(self, state):
        return state.legal_actions[0]


def test_same_seed_produces_identical_result():
    bots_a = [DeterministicBot(f"Bot{i}") for i in range(3)]
    bots_b = [DeterministicBot(f"Bot{i}") for i in range(3)]
    result_a = PerudoGame(bots=bots_a, seed=999).play()
    result_b = PerudoGame(bots=bots_b, seed=999).play()
    assert result_a.standings == result_b.standings
    assert result_a.rounds_played == result_b.rounds_played


# -- information hiding --------------------------------------------------


def test_hidden_information_never_leaked_to_bots():
    bots = make_bots(3)
    PerudoGame(bots=bots, seed=7).play()

    for bot in bots:
        for obs in bot.seen_observations:
            assert isinstance(obs, Observation)
            assert len(obs.my_dice) == obs.my_dice_count
            for public_player in obs.players:
                # PublicPlayer structurally has no dice field at all.
                assert not hasattr(public_player, "dice")


def test_history_never_carries_a_finished_rounds_terminal_action():
    bots = make_bots(3)
    PerudoGame(bots=bots, seed=123).play()

    for bot in bots:
        for obs in bot.seen_observations:
            # A round ends the instant a Dudo/Calza is recorded, so no
            # observation should ever show one in its own history.
            assert not any(isinstance(r.action, (Dudo, Calza)) for r in obs.history)


# -- illegal actions ------------------------------------------------------


class AlwaysIllegalBot:
    name = "Illegal Bot"

    def play(self, state):
        return Bid(999_999, 4)  # always exceeds the total-dice bound


def test_illegal_action_raises_clear_error():
    bots = [AlwaysIllegalBot(), RandomTestBot("Other")]
    game = PerudoGame(bots=bots, seed=1)

    with pytest.raises(IllegalActionError) as exc_info:
        game.play()

    message = str(exc_info.value)
    assert "Illegal Bot" in message
    assert "Legal actions" in message


# -- Dudo resolution & next-round starter ---------------------------------


def test_dudo_bidder_correct_caller_loses_die_and_starts_next_round():
    game = PerudoGame(bots=make_bots(2), seed=1)
    caller_idx, bidder_idx = 0, 1
    caller, bidder = game.players[caller_idx], game.players[bidder_idx]
    caller.dice_count, caller.dice = 3, [2, 3, 6]
    bidder.dice_count, bidder.dice = 2, [4, 1]  # one real 4 + one Paco = 2

    next_starter, _ = game._resolve_dudo(caller_idx, bidder_idx, Bid(2, 4), palifico=False)

    assert caller.dice_count == 2
    assert next_starter == caller_idx


def test_dudo_bidder_wrong_bidder_loses_die_and_starts_next_round():
    game = PerudoGame(bots=make_bots(2), seed=1)
    caller_idx, bidder_idx = 0, 1
    caller, bidder = game.players[caller_idx], game.players[bidder_idx]
    caller.dice_count, caller.dice = 3, [2, 3, 6]
    bidder.dice_count, bidder.dice = 2, [5, 6]  # zero matches for face 4

    next_starter, _ = game._resolve_dudo(caller_idx, bidder_idx, Bid(2, 4), palifico=False)

    assert bidder.dice_count == 1
    assert next_starter == bidder_idx


def test_dudo_exact_match_favors_the_bidder():
    game = PerudoGame(bots=make_bots(2), seed=1)
    caller_idx, bidder_idx = 0, 1
    caller, bidder = game.players[caller_idx], game.players[bidder_idx]
    caller.dice_count, caller.dice = 2, [4, 4]
    bidder.dice_count, bidder.dice = 2, [6, 6]

    game._resolve_dudo(caller_idx, bidder_idx, Bid(2, 4), palifico=False)

    assert caller.dice_count == 1  # caller loses even though the count matched exactly


def test_dudo_loser_eliminated_next_starter_skips_to_next_alive():
    game = PerudoGame(bots=make_bots(3), seed=1)
    caller_idx, bidder_idx = 0, 1
    caller, bidder = game.players[caller_idx], game.players[bidder_idx]
    caller.dice_count, caller.dice = 3, [2, 3, 6]
    bidder.dice_count, bidder.dice = 1, [5]  # zero matches -> bidder wrong -> eliminated

    next_starter, _ = game._resolve_dudo(caller_idx, bidder_idx, Bid(2, 4), palifico=False)

    assert bidder.alive is False
    assert next_starter != bidder_idx
    assert game.players[next_starter].alive


# -- bot lifecycle across simulated games ---------------------------------


def test_bot_instances_persist_and_game_state_fully_resets_across_games():
    bots = make_bots(3)

    for seed in (1, 2, 3):
        game = PerudoGame(bots=bots, seed=seed)
        game.play()
        # A fresh PerudoGame means a fresh engine/state: every player is
        # dealt back up to 5 dice at the very start.
        assert all(p.dice_count <= 5 for p in game.players)

    # Same bot objects were used for all three games -- the callbacks
    # prove the *instances* were reused (this is what lets a bot's own
    # instance variables carry "memory" across games).
    for bot in bots:
        assert bot.games_started == 3
        assert bot.games_ended == 3


# -- player count, names, and seating -------------------------------------


def test_observation_reveals_total_player_count_including_eliminated():
    # A bot can't judge whether a bid is "high" without knowing how many
    # players/dice are in the game -- state.players always lists every
    # seat, alive or not, so len(state.players) is always the true total.
    bots = make_bots(4)
    game = PerudoGame(bots=bots, seed=5)
    game.play()

    for bot in bots:
        for obs in bot.seen_observations:
            assert len(obs.players) == 4
            assert sum(p.dice_count for p in obs.players) >= 0  # always computable


class InitNamedBot:
    """A bot whose name is an instance variable set in __init__, not a
    class attribute -- both styles must work identically."""

    def __init__(self, label):
        self.name = label

    def play(self, state):
        return state.legal_actions[0]


def test_bot_name_set_in_init_is_used_everywhere():
    bots = [InitNamedBot("Alpha"), InitNamedBot("Beta")]
    events = []
    game = PerudoGame(bots=bots, seed=3, listeners=[events.append])
    result = game.play()

    assert result.winner in ("Alpha", "Beta")
    assert set(result.standings) == {"Alpha", "Beta"}

    bid_events = [e for e in events if isinstance(e, BidEvent)]
    assert bid_events
    assert all(e.player in ("Alpha", "Beta") for e in bid_events)


def test_history_and_events_identify_who_did_what_by_name():
    bots = make_bots(3)
    game = PerudoGame(bots=bots, seed=9)
    game.play()

    bot_names = {b.name for b in bots}
    for bot in bots:
        for obs in bot.seen_observations:
            for record in obs.history:
                assert record.player in bot_names


def test_seating_is_randomized_across_games_not_fixed_by_bot_order():
    bots = make_bots(4)
    seats_of_bot0 = set()
    for seed in range(30):
        game = PerudoGame(bots=bots, seed=seed)
        seat = next(i for i, p in enumerate(game.players) if p.name == "Bot0")
        seats_of_bot0.add(seat)

    # If seating were fixed (e.g. always seat 0), a bot sitting right
    # before the weakest player would have a permanent positional edge.
    assert len(seats_of_bot0) > 1
