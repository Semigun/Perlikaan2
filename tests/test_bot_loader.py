"""
Tests for the `count` class-attribute mechanism (multiple copies of the
same bot, auto-numbered) and the --bot CLI lineup builder.
"""

import perudo.bot_loader as bot_loader


class FakeBot:
    name = "Fake Bot"

    def play(self, state):
        return state.legal_actions[0]


class TripleBotClassAttr:
    name = "Triple Bot"
    count = 3

    def play(self, state):
        return state.legal_actions[0]


class TripleBotInitAttr:
    def __init__(self):
        self.name = "Triple Init Bot"
        self.count = 3

    def play(self, state):
        return state.legal_actions[0]


def test_numbered_instantiation_leaves_first_name_untouched():
    bots = bot_loader._instantiate_with_copies(TripleBotClassAttr)
    assert [b.name for b in bots] == ["Triple Bot", "Triple Bot 2", "Triple Bot 3"]


def test_load_bots_respects_count_class_attribute(monkeypatch):
    monkeypatch.setattr(bot_loader, "discover_bot_classes", lambda: [TripleBotClassAttr])
    bots = bot_loader.load_bots()
    assert [b.name for b in bots] == ["Triple Bot", "Triple Bot 2", "Triple Bot 3"]


def test_load_bots_respects_count_set_in_init(monkeypatch):
    # self.count = N set inside __init__ (alongside self.name) must work
    # just as well as a plain class-level `count = N`.
    monkeypatch.setattr(bot_loader, "discover_bot_classes", lambda: [TripleBotInitAttr])
    bots = bot_loader.load_bots()
    assert [b.name for b in bots] == ["Triple Init Bot", "Triple Init Bot 2", "Triple Init Bot 3"]


def test_load_bots_defaults_to_count_one_when_absent(monkeypatch):
    monkeypatch.setattr(bot_loader, "discover_bot_classes", lambda: [FakeBot])
    bots = bot_loader.load_bots()
    assert [b.name for b in bots] == ["Fake Bot"]


class BenchedBot:
    def __init__(self):
        self.name = "Benched Bot"
        self.count = 0

    def play(self, state):
        return state.legal_actions[0]


def test_load_bots_excludes_bot_with_count_zero(monkeypatch):
    monkeypatch.setattr(bot_loader, "discover_bot_classes", lambda: [BenchedBot, FakeBot])
    bots = bot_loader.load_bots()
    assert [b.name for b in bots] == ["Fake Bot"]


def test_instantiate_bots_by_name_numbers_duplicates(monkeypatch):
    monkeypatch.setattr(
        bot_loader, "discover_bot_classes_by_name", lambda: {"FakeBot": FakeBot}
    )
    bots = bot_loader.instantiate_bots(["FakeBot", "FakeBot"])
    assert [b.name for b in bots] == ["Fake Bot", "Fake Bot 2"]
