"""
Discovers bot classes under perudo/bots/ and instantiates them.

A participant only has to drop a new .py file into perudo/bots/ with a
class that has a `play` method -- both run_interactive.py and
run_simulation.py automatically pick it up. No registration step needed.

Want multiple copies of the same bot at the table (e.g. to see how it
does against itself)? Just add a `count`, either as a class attribute or
as `self.count` in `__init__` (both work):

    class RandomBot:
        name = "Random Bot"
        count = 3   # play with 3 copies of this bot

        def play(self, state):
            ...

The extra copies are auto-numbered ("Random Bot 2", "Random Bot 3", ...)
so every bot still has the unique name PerudoGame requires -- no need to
copy the file and hand-edit a name. Set `count = 0` to bench a bot (it
won't be instantiated at all) without deleting its file.
"""

from __future__ import annotations

import importlib.util
import inspect
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

BOTS_DIR = Path(__file__).parent / "bots"


def discover_bot_classes() -> List[type]:
    bot_classes: List[type] = []
    for path in sorted(BOTS_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue

        module_name = f"perudo.bots.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ != module.__name__:
                continue  # imported from elsewhere -- not a bot defined here
            if not hasattr(obj, "play"):
                continue
            bot_classes.append(obj)
    return bot_classes


def discover_bot_classes_by_name() -> Dict[str, type]:
    return {cls.__name__: cls for cls in discover_bot_classes()}


def _instantiate_with_copies(cls: type) -> List[Any]:
    """
    Instantiate `cls` once, then check that instance's `count` (read after
    construction, so `self.count = N` set in __init__ is honored just as
    well as a plain `count = N` class attribute) to make that many extra
    numbered copies ("Name 2", "Name 3", ...). `count = 0` excludes the
    bot entirely -- a handy way to bench a bot without deleting its file.
    """
    first = cls()
    count = getattr(first, "count", 1)
    if count <= 0:
        return []
    bots = [first]
    for i in range(2, count + 1):
        extra = cls()
        extra.name = f"{extra.name} {i}"
        bots.append(extra)
    return bots


def load_bots() -> List[Any]:
    """
    Instantiate every bot class found under perudo/bots/.

    Normally exactly one instance per class -- unless the bot has a
    `count` of more than 1 (see module docstring), in which case it is
    instantiated that many times.
    """
    bots: List[Any] = []
    for cls in discover_bot_classes():
        bots.extend(_instantiate_with_copies(cls))
    return bots


def instantiate_bots(class_names: List[str]) -> List[Any]:
    """
    Build a specific lineup by class name, e.g. ["RandomBot", "RandomBot",
    "ExampleBot"] for two RandomBots against one ExampleBot. Used by the
    --bot CLI flag; class-level `count` (see load_bots()) is simpler for
    everyday use and is what --bot-less runs use.
    """
    classes_by_name = discover_bot_classes_by_name()
    requested_counts = Counter(class_names)
    seen_so_far: Counter = Counter()

    bots: List[Any] = []
    for class_name in class_names:
        if class_name not in classes_by_name:
            available = ", ".join(sorted(classes_by_name)) or "(geen bots gevonden)"
            raise ValueError(
                f"Onbekende bot-class '{class_name}'. Beschikbaar: {available}"
            )
        bot = classes_by_name[class_name]()
        if requested_counts[class_name] > 1:
            seen_so_far[class_name] += 1
            if seen_so_far[class_name] > 1:
                bot.name = f"{bot.name} {seen_so_far[class_name]}"
        bots.append(bot)
    return bots
