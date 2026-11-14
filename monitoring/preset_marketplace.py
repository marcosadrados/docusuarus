"""Presets marketplace for the scalping trading bot.

A catalog of named strategy presets that can be pulled in by name and used
straight by the engine.  Presets are seeded through :mod:`signals.registry`
(the built-in presets) and, for rules, built from the no-code rule DSL in
``signals.rules``.

The marketplace is driven by an external catalog file (``docs/strategy_cards.json``)
so local presets (stat-arb, regime, ensemble, freqai) can be added or their
evidence updated without touching Python source.  When the catalog file is
missing or unreadable the marketplace falls back to the built-in set.

Built-in presets
----------------
- market_trend          : EMA momentum + mean-reversion (cTrader / crypto)
- rule_momentum         : rule DSL, e.g. ``close > ema(5) and volume > 1e6``
- rule_mean_reversion   : rule DSL, e.g. ``rsi(14) > 70``

Usage
-----
>>> from monitoring.preset_marketplace import PresetMarketplace
>>> pm = PresetMarketplace()
>>> strat = pm.load_preset("rule_momentum")
>>> strat is not None
True
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from signals.freqai import Freqai
from signals.regime import RegimeSwitcher, RegimeStrategy
from signals.registry import StrategyRegistry, load_strategy_preset
from signals.rules import RuleStrategy, parse_rules

__all__ = ["PresetMarketplace"]

#: Where the external catalog lives.  One level up from this package.
_DEFAULT_CATALOG = Path(__file__).resolve().parent.parent / "docs" / "strategy_cards.json"

#: Builder names accepted in an external (JSON) catalog entry.  Built-ins keep
#: their callable ``build`` directly.
def _build_market_trend() -> Any:
    return load_strategy_preset("market_trend")


def _build_rule(name: str, dsl: str) -> RuleStrategy:
    return RuleStrategy(parse_rules(dsl))


def _build_stat_arb() -> Any:
    return load_strategy_preset("pairs_stat_arb")


def _build_regime() -> Any:
    return RegimeStrategy(RegimeSwitcher())


def _build_ensemble() -> Any:
    return load_strategy_preset("signal_ensemble")


def _build_freqai() -> Any:
    return Freqai()


_BUILTIN_PRESETS: Dict[str, Dict[str, Any]] = {
    "market_trend": {
        "title": "EMA momentum + mean reversion",
        "build": _build_market_trend,
        "description": (
            "Momentum: close > ema(5) and volume > p50; mean-reversion: "
            "close > 2*std of 50-ma."
        ),
    },
    "rule_momentum": {
        "title": "Rule: momentum scalper (BUY)",
        "build": lambda: _build_rule(
            "rule_momentum", "if close > ema(5) and volume > 1000000: BUY"
        ),
        "description": "Buys on momentum; pairs well with the risk scaler.",
    },
    "rule_mean_reversion": {
        "title": "Rule: mean reversion (SELL)",
        "build": lambda: _build_rule(
            "rule_mean_reversion", "if rsi(14) > 70: SELL"
        ),
        "description": "Sells when RSI(14) is overbought.",
    },
}

#: External-catalog builder names -> callables.  Resolved at load time so a
#: JSON catalog never has to ship Python callables.
_BUILDER_REGISTRY: Dict[str, Any] = {
    "market_trend": _build_market_trend,
    "stat_arb": _build_stat_arb,
    "regime_adaptive": _build_regime,
    "signal_ensemble": _build_ensemble,
    "freqai": _build_freqai,
    "rule": lambda name="rule", dsl="if close > ema(5): BUY": _build_rule(name, dsl),
}


class PresetMarketplace:
    """Catalog + loader for named strategy presets."""

    def __init__(
        self, registry: Optional[StrategyRegistry] = None, catalog_path: Optional[Path] = None
    ) -> None:
        self.registry = registry or StrategyRegistry()
        self.catalog_path = catalog_path or _DEFAULT_CATALOG
        # Start from the built-ins, then overlay the external catalog so JSON
        # entries add or update presets without code changes.
        self.catalog = {name: dict(spec) for name, spec in _BUILTIN_PRESETS.items()}
        self._merge_external(self._load_external_catalog())
        self.deployed: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Catalog access
    # ------------------------------------------------------------------

    def available(self) -> list[str]:
        return sorted(self.catalog)

    def describe(self, name: str) -> Optional[Dict[str, Any]]:
        return dict(self.catalog.get(name, {}))

    # ------------------------------------------------------------------
    # Loading / installing
    # ------------------------------------------------------------------

    def load_preset(self, name: str, **kwargs: Any) -> Any:
        """Build a ready-to-use strategy instance from a preset name.

        ``build`` may be a callable (built-in) or a string resolved through
        :attr:`_BUILDER_REGISTRY` (external catalog).
        """
        if name not in self.catalog:
            raise ValueError(
                f"unknown preset: {name!r}; available: {self.available()}"
            )
        spec = self.catalog[name]
        build = spec.get("build")
        if isinstance(build, str):
            build = self._BUILDER_REGISTRY[build]
        return build(**kwargs)

    def register_preset(self, name: str, title: str, build: Any) -> None:
        """Register a user-supplied preset at runtime."""
        if not name or not build:
            raise ValueError("preset name and build callable are required")
        if name in self.catalog:
            raise ValueError(f"preset {name!r} already exists")
        self.catalog[name] = {"title": title, "build": build}

    def install(self, name: str, **kwargs: Any) -> Any:
        """Load and deploy a preset (records it in :attr:`deployed`)."""
        instance = self.load_preset(name, **kwargs)
        self.deployed[name] = instance
        return instance

    # ------------------------------------------------------------------
    # Internal: external catalog
    # ------------------------------------------------------------------

    def _load_external_catalog(self) -> Dict[str, Any]:
        """Load JSON strategy cards if present; empty dict otherwise."""
        try:
            if not self.catalog_path.exists():
                return {}
            data = json.loads(self.catalog_path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return {item.get("name"): item for item in data if isinstance(item, dict)}
        except Exception:
            pass
        return {}

    def _merge_external(self, external: Dict[str, Any]) -> None:
        for name, entry in external.items():
            # Only runnable cards (those that declare a ``build``) are merged
            # into the runtime loader; descriptive-only cards (e.g. a placeholder
            # rule) stay in the JSON card index but are not loadable here.
            if "build" not in entry and not entry.get("loadable", False):
                continue
            self.catalog[name] = self._coerce_entry(entry)

    @staticmethod
    def _coerce_entry(entry: Dict[str, Any]) -> Dict[str, Any]:
        spec = dict(entry)
        # Build may be a callable (from a local dict) or a string (from JSON);
        # normalise to a callable now so callers never resolve it twice.
        build = spec.get("build")
        if isinstance(build, str):
            spec["build"] = _BUILDER_REGISTRY[build]
        elif build is None:
            spec["build"] = _BUILDER_REGISTRY.get(spec.get("builder") or "rule")
        return spec
