"""
Interactive mode: play a single Perudo game in the terminal, one ENTER
press per turn. Bots are initialized exactly once (this is a single game,
so that doesn't matter much), and there is no memory between runs.

Usage:
    python run_interactive.py --seed 938271

This file contains NO game rules -- it only renders events produced by the
engine (perudo.game.PerudoGame) and pauses for ENTER at the right moments.
"""

from __future__ import annotations

import argparse
import random

from perudo.bot_loader import instantiate_bots, load_bots
from perudo.game import PerudoGame
from perudo.models import (
    BidEvent,
    CalzaEvent,
    DieGainedEvent,
    DieLostEvent,
    DudoEvent,
    GameEndedEvent,
    PlayerEliminatedEvent,
    RevealEvent,
    RoundEndedEvent,
)


def format_bid(bid) -> str:
    face = "Paco" if bid.face == 1 else str(bid.face)
    return f"{bid.quantity}x {face}"


class InteractiveRenderer:
    def __init__(self):
        self.last_printed_round = 0

    def before_turn(self, obs) -> None:
        if obs.round_number != self.last_printed_round:
            self.last_printed_round = obs.round_number
            print(f"\n{'=' * 50}")
            print(f"Ronde {obs.round_number}   |   Palifico: {'JA' if obs.palifico else 'NEE'}")
            print(f"{'=' * 50}")
            for p in obs.players:
                status = "actief" if p.alive else "uitgeschakeld"
                print(f"  {p.name}: {p.dice_count} dobbelstenen ({status})")

        me = obs.players[obs.my_id]
        print(f"\n{me.name}'s beurt.")
        input("Druk op ENTER...")

    def before_reveal(self, challenge_type, challenger, bidder, current_bid) -> None:
        label = "DUDO" if challenge_type == "dudo" else "CALZA"
        print(f"\n{challenger} roept {label}!")
        input("Druk op ENTER om de dobbelstenen te onthullen...")

    def on_event(self, event) -> None:
        if isinstance(event, BidEvent):
            print(f"  -> {event.player} biedt {format_bid(event.bid)}")
        elif isinstance(event, DudoEvent):
            pass  # announced in before_reveal
        elif isinstance(event, CalzaEvent):
            pass  # announced in before_reveal
        elif isinstance(event, RevealEvent):
            print("\n  Dobbelstenen op tafel:")
            for name, dice in event.dice:
                print(f"    {name}: {dice}")
            print(
                f"  Bod was {format_bid(event.current_bid)} -> "
                f"werkelijk aantal: {event.actual_count}"
            )
            if event.challenge_type == "dudo":
                if event.result == "bidder_correct":
                    print(f"  Bod klopte (of meer) -> {event.challenger} verliest een dobbelsteen.")
                else:
                    print(f"  Bod klopte niet -> {event.bidder} verliest een dobbelsteen.")
            else:
                if event.result == "correct":
                    print(f"  Calza is EXACT correct!")
                else:
                    print(f"  Calza is FOUT -> {event.challenger} verliest een dobbelsteen.")
        elif isinstance(event, DieLostEvent):
            print(f"  {event.player} heeft nu nog {event.dice_count} dobbelsteen(en).")
        elif isinstance(event, DieGainedEvent):
            print(f"  {event.player} krijgt een dobbelsteen terug! Nu {event.dice_count}.")
        elif isinstance(event, PlayerEliminatedEvent):
            print(f"  *** {event.player} is UITGESCHAKELD! ***")
        elif isinstance(event, RoundEndedEvent):
            if event.palifico_next:
                print(f"\n  Volgende ronde is PALIFICO! {event.next_starting_player} begint.")
            else:
                print(f"\n  {event.next_starting_player} begint de volgende ronde.")
        elif isinstance(event, GameEndedEvent):
            print(f"\n{'#' * 50}")
            print(f"  WINNAAR: {event.winner}")
            print(f"  Eindstand: {', '.join(event.standings)}")
            print(f"{'#' * 50}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Perudo interactive mode (1 game, press ENTER per turn).")
    parser.add_argument("--seed", type=int, default=None, help="Seed voor reproduceerbaarheid.")
    parser.add_argument(
        "--bot",
        action="append",
        dest="bots",
        default=None,
        metavar="CLASSNAME",
        help=(
            "Botclass om mee te laten spelen (herhaal voor meerdere spelers), "
            "bv. --bot RandomBot --bot RandomBot --bot ExampleBot --bot ExampleBot. "
            "Zonder deze vlag speelt elke bot in perudo/bots/ precies één keer mee."
        ),
    )
    args = parser.parse_args()

    seed = args.seed if args.seed is not None else random.SystemRandom().randrange(2**63)

    bots = instantiate_bots(args.bots) if args.bots else load_bots()

    print("PERUDO BOT BATTLE")
    for bot in bots:
        print(f"  {bot.name} - 5 dice")
    print(f"\n(seed = {seed})")

    renderer = InteractiveRenderer()
    game = PerudoGame(
        bots=bots,
        seed=seed,
        before_turn=renderer.before_turn,
        before_reveal=renderer.before_reveal,
        listeners=[renderer.on_event],
    )
    game.play()


if __name__ == "__main__":
    main()
