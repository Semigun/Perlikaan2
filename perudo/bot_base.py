"""
Optional base class for bots.

A bot does NOT have to inherit from BaseBot -- plain duck typing works fine
(see bots/random_bot.py, which only implements `play`). Inheriting from
BaseBot just saves you from having to write empty stub methods for the
callbacks you don't care about.
"""

from __future__ import annotations

from perudo.models import Action, Event, GameResult, GameStartInfo, Observation


class BaseBot:
    name: str = "Unnamed Bot"

    def on_game_start(self, info: GameStartInfo) -> None:
        """Called once per game, before the first round. Optional."""

    def play(self, state: Observation) -> Action:
        """Called on your turn. Must return one action from state.legal_actions."""
        raise NotImplementedError("Implement play() in your bot.")

    def observe(self, event: Event) -> None:
        """Called for every public event in the game. Optional."""

    def on_game_end(self, result: GameResult) -> None:
        """Called once when the game is over. Optional."""
