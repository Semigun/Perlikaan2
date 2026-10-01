"""
Pure bidding-rule tests. These test perudo.rules directly -- no game
engine, no players, no randomness -- so they pin down the bidding math in
complete isolation.
"""

from perudo.models import PACO, Bid, Calza, Dudo
from perudo.rules import count_matches, legal_actions, legal_bids


def test_opening_bid_normal_round_excludes_paco():
    bids = legal_bids(current_bid=None, palifico=False, palifico_face=None, total_dice=10)
    assert all(b.face != PACO for b in bids)
    assert Bid(1, 2) in bids
    assert Bid(10, 6) in bids


def test_opening_bid_palifico_round_allows_paco():
    bids = legal_bids(current_bid=None, palifico=True, palifico_face=None, total_dice=10)
    assert any(b.face == PACO for b in bids)
    assert any(b.face != PACO for b in bids)


def test_normal_raise_same_quantity_higher_face_only():
    bids = legal_bids(Bid(5, 4), palifico=False, palifico_face=None, total_dice=20)
    assert Bid(5, 5) in bids
    assert Bid(5, 6) in bids
    assert Bid(5, 2) not in bids
    assert Bid(5, 3) not in bids
    assert Bid(5, 4) not in bids  # must be strictly higher


def test_normal_raise_higher_quantity_allows_any_normal_face():
    bids = legal_bids(Bid(5, 4), palifico=False, palifico_face=None, total_dice=20)
    for face in (2, 3, 4, 5, 6):
        assert Bid(6, face) in bids


def test_switch_to_paco_minimum_quantity_is_ceil_half():
    # (7, 4) -> minimum (4, Paco)
    bids = legal_bids(Bid(7, 4), palifico=False, palifico_face=None, total_dice=20)
    assert Bid(4, PACO) in bids
    assert Bid(3, PACO) not in bids


def test_switch_from_paco_to_normal_minimum_quantity_is_double_plus_one():
    # (4, Paco) -> minimum (9, 2..6)
    bids = legal_bids(Bid(4, PACO), palifico=False, palifico_face=None, total_dice=20)
    assert Bid(9, 2) in bids
    assert Bid(9, 6) in bids
    assert Bid(8, 2) not in bids


def test_paco_to_paco_must_strictly_increase_quantity():
    bids = legal_bids(Bid(4, PACO), palifico=False, palifico_face=None, total_dice=20)
    paco_bids = [b for b in bids if b.face == PACO]
    assert Bid(5, PACO) in paco_bids
    assert Bid(4, PACO) not in paco_bids
    assert all(b.quantity > 4 for b in paco_bids)


def test_palifico_face_is_locked_after_opening_bid():
    bids = legal_bids(Bid(2, 4), palifico=True, palifico_face=4, total_dice=20)
    assert all(b.face == 4 for b in bids)
    assert Bid(3, 4) in bids
    assert Bid(3, 5) not in bids
    assert Bid(2, 4) not in bids  # must strictly increase


def test_palifico_opening_bid_can_be_paco_and_then_locks_it():
    bids = legal_bids(Bid(1, PACO), palifico=True, palifico_face=PACO, total_dice=20)
    assert all(b.face == PACO for b in bids)
    assert Bid(2, PACO) in bids


def test_dudo_and_calza_available_once_a_bid_exists():
    actions = legal_actions(Bid(3, 4), palifico=False, palifico_face=None, total_dice=20)
    assert Dudo() in actions
    assert Calza() in actions


def test_no_dudo_or_calza_on_opening_bid():
    actions = legal_actions(None, palifico=False, palifico_face=None, total_dice=20)
    assert Dudo() not in actions
    assert Calza() not in actions


def test_count_matches_wild_paco_in_normal_round():
    # two real 4s + two Pacos count towards a bid on 4
    assert count_matches([1, 1, 4, 4, 5], face=4, palifico=False) == 4


def test_count_matches_paco_bid_counts_only_paco():
    assert count_matches([1, 1, 4, 4, 5], face=PACO, palifico=False) == 2


def test_count_matches_no_wild_during_palifico():
    assert count_matches([1, 1, 4, 4, 5], face=4, palifico=True) == 2
    assert count_matches([1, 1, 4, 4, 5], face=PACO, palifico=True) == 2
