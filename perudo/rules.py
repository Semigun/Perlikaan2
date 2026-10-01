"""
Pure Perudo rule functions: legal bid enumeration and counting.

These functions take plain values in and return plain values out -- no
game/engine object involved -- so the rules can be unit tested in complete
isolation from turn order, dice rolling, players, etc.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence

from perudo.models import PACO, NORMAL_FACES, Action, Bid, Calza, Dudo


def count_matches(dice_values: Sequence[int], face: int, palifico: bool) -> int:
    """How many dice count towards a bid on `face`."""
    if palifico:
        return sum(1 for d in dice_values if d == face)
    if face == PACO:
        return sum(1 for d in dice_values if d == PACO)
    return sum(1 for d in dice_values if d == face or d == PACO)


def legal_bids(
    current_bid: Optional[Bid],
    palifico: bool,
    palifico_face: Optional[int],
    total_dice: int,
) -> List[Bid]:
    """All legal Bid actions given the current state of a round."""
    max_q = max(total_dice, 1)
    bids: List[Bid] = []

    if current_bid is None:
        # Opening bid of the round.
        faces = range(1, 7) if palifico else NORMAL_FACES
        for face in faces:
            for q in range(1, max_q + 1):
                bids.append(Bid(q, face))
        return bids

    q0, f0 = current_bid.quantity, current_bid.face

    if palifico:
        # Face is locked for the whole round; only quantity may increase.
        face = palifico_face
        for q in range(q0 + 1, max_q + 1):
            bids.append(Bid(q, face))
        return bids

    # Normal (non-Palifico) round, a bid already exists.
    if f0 == PACO:
        # Paco -> higher Paco.
        for q in range(q0 + 1, max_q + 1):
            bids.append(Bid(q, PACO))
        # Paco -> normal face, quantity must at least double (+1).
        min_q_normal = 2 * q0 + 1
        for q in range(min_q_normal, max_q + 1):
            for face in NORMAL_FACES:
                bids.append(Bid(q, face))
    else:
        # Same quantity, strictly higher normal face.
        for face in range(f0 + 1, 7):
            bids.append(Bid(q0, face))
        # Higher quantity, any normal face (including the same one).
        for q in range(q0 + 1, max_q + 1):
            for face in NORMAL_FACES:
                bids.append(Bid(q, face))
        # Normal -> Paco, quantity at least ceil(q0 / 2).
        min_q_paco = math.ceil(q0 / 2)
        for q in range(min_q_paco, max_q + 1):
            bids.append(Bid(q, PACO))

    return bids


def legal_actions(
    current_bid: Optional[Bid],
    palifico: bool,
    palifico_face: Optional[int],
    total_dice: int,
) -> List[Action]:
    """All legal actions (bids, plus Dudo/Calza once a bid exists)."""
    actions: List[Action] = list(
        legal_bids(current_bid, palifico, palifico_face, total_dice)
    )
    if current_bid is not None:
        actions.append(Dudo())
        actions.append(Calza())
    return actions
