"""Tests for the presets marketplace."""

import pytest

from monitoring.preset_marketplace import PresetMarketplace
from signals.rules import RuleStrategy
from signals.strategy import MarketTrendStrategy


def test_available_preset_names():
    pm = PresetMarketplace()
    names = pm.available()
    assert "market_trend" in names
    assert "rule_momentum" in names
    assert "rule_mean_reversion" in names


def test_describe_preset():
    pm = PresetMarketplace()
    desc = pm.describe("rule_momentum")
    assert desc is not None
    assert "momentum" in desc["title"].lower()
    assert desc["title"]


def test_load_preset_market_trend():
    pm = PresetMarketplace()
    strat = pm.load_preset("market_trend")
    assert isinstance(strat, MarketTrendStrategy)


def test_load_preset_rule_momentum():
    pm = PresetMarketplace()
    strat = pm.load_preset("rule_momentum")
    assert isinstance(strat, RuleStrategy)
    assert len(strat) == 1
    assert strat.rules[0].direction == "BUY"


def test_load_preset_rule_mean_reversion():
    pm = PresetMarketplace()
    strat = pm.load_preset("rule_mean_reversion")
    assert isinstance(strat, RuleStrategy)
    assert len(strat) == 1
    assert strat.rules[0].direction == "SELL"


def test_load_preset_unknown_raises():
    pm = PresetMarketplace()
    with pytest.raises(ValueError):
        pm.load_preset("not_a_preset")


def test_register_preset():
    pm = PresetMarketplace()
    built = []

    def dummy_build():
        built.append(1)
        return object()

    pm.register_preset("my_preset", "A dummy preset", dummy_build)
    assert "my_preset" in pm.available()
    inst = pm.load_preset("my_preset")
    assert inst is not None
    assert len(built) == 1

    # Built-ins AND JSON-catalog presets resolve through the same loader.
    assert pm.load_preset("market_trend") is not None
    assert pm.load_preset("stat_arb") is not None
    assert pm.load_preset("freqai") is not None
    assert pm.load_preset("regime_adaptive") is not None


def test_install_deploys():
    pm = PresetMarketplace()
    inst = pm.install("market_trend")
    assert inst in pm.deployed.values()


# ------------------------------------------------------------------
# Wiring: Orchestrator reads strategy from config
# ------------------------------------------------------------------


def test_orchestrator_loads_rule_preset_from_config():
    from orchestrator.main import Orchestrator

    orch = Orchestrator(config={"strategy": "rule_momentum", "log_level": "ERROR"})
    assert isinstance(orch.strategy, RuleStrategy)
    assert orch.risk is not None
