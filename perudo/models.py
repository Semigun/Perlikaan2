"""
Core data types for the Perudo engine.

Everything here is a plain, frozen (immutable) dataclass or a tiny value
type. Bots only ever see these types -- never the internal, mutable game
state. Because every nested object is itself frozen/tuple-based, there is
no way for a bot to reach back into the engine and mutate (or leak) hidden
information through a shared reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

PACO = 1
NORMAL_FACES = (2, 3, 4, 5, 6)


# ---------------------------------------------------------------------------
# Actions a bot can return from play()
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Bid:
    quantity: int
    face: int  # 1..6, where 1 means Paco

    def __repr__(self) -> str:
        face = "Paco" if self.face == PACO else str(self.face)
        return f"Bid({self.quantity}x {face})"


@dataclass(frozen=True)
class Dudo:
    def __repr__(self) -> str:
        return "Dudo()"


@dataclass(frozen=True)
class Calza:
    def __repr__(self) -> str:
        return "Calza()"


Action = Union[Bid, Dudo, Calza]


# ---------------------------------------------------------------------------
# Public information
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionRecord:
    """One entry in the current round's history."""

    player: str
    action: Action


@dataclass(frozen=True)
class PublicPlayer:
    """Publicly known information about a player (never their hidden dice)."""

    id: int
    name: str
    dice_count: int
    alive: bool


@dataclass(frozen=True)
class Observation:
    """
    The only thing a bot ever sees. Immutable snapshot of "what I'm allowed
    to know right now". Never hands out another player's dice.
    """

    my_id: int
    my_dice: Tuple[int, ...]
    my_dice_count: int

    players: Tuple[PublicPlayer, ...]
    current_bid: Optional[Bid]
    palifico: bool
    round_number: int

    history: Tuple[ActionRecord, ...]
    legal_actions: Tuple[Action, ...]


# ---------------------------------------------------------------------------
# Lifecycle callbacks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GameStartInfo:
    players: Tuple[PublicPlayer, ...]
    my_id: int


@dataclass(frozen=True)
class GameResult:
    winner: str
    standings: Tuple[str, ...]  # winner first, last-place last
    rounds_played: int


# ---------------------------------------------------------------------------
# Public events (delivered to every bot via observe(), and to the
# interactive renderer / simulation statistics via listeners)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BidEvent:
    player: str
    bid: Bid


@dataclass(frozen=True)
class DudoEvent:
    player: str


@dataclass(frozen=True)
class CalzaEvent:
    player: str


@dataclass(frozen=True)
class RevealEvent:
    """Everyone's dice become public information after a Dudo/Calza."""

    dice: Tuple[Tuple[str, Tuple[int, ...]], ...]  # (player_name, dice) pairs
    current_bid: Bid
    actual_count: int
    challenge_type: str  # "dudo" or "calza"
    challenger: str
    bidder: str
    result: str  # dudo: "bidder_correct" / "bidder_wrong"; calza: "correct" / "wrong"

    def dice_by_player(self) -> dict:
        return dict(self.dice)


@dataclass(frozen=True)
class DieLostEvent:
    player: str
    dice_count: int  # new count, after losing the die


@dataclass(frozen=True)
class DieGainedEvent:
    player: str
    dice_count: int  # new count, after gaining the die


@dataclass(frozen=True)
class PlayerEliminatedEvent:
    player: str


@dataclass(frozen=True)
class RoundEndedEvent:
    next_starting_player: str
    palifico_next: bool


@dataclass(frozen=True)
class GameEndedEvent:
    winner: str
    standings: Tuple[str, ...]


Event = Union[
    BidEvent,
    DudoEvent,
    CalzaEvent,
    RevealEvent,
    DieLostEvent,
    DieGainedEvent,
    PlayerEliminatedEvent,
    RoundEndedEvent,
    GameEndedEvent,
]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class IllegalActionError(Exception):
    """Raised when a bot returns an action that is not in state.legal_actions."""

    def __init__(self, bot_name: str, action: object, current_bid: Optional[Bid], legal_actions):
        legal_list = list(legal_actions)
        message = (
            f"Bot '{bot_name}' returned an illegal action.\n"
            f"  Returned action : {action!r}\n"
            f"  Current bid     : {current_bid!r}\n"
            f"  Legal actions   : {legal_list!r}"
        )
        super().__init__(message)
        self.bot_name = bot_name
        self.action = action
        self.current_bid = current_bid
        self.legal_actions = legal_list
