"""Strategy registry + preset loader for the scalping trading bot.

The engine (``orchestrator.engine.Engine``) is strategy-agnostic: it only
needs an object with ``generate(candles) -> list[Signal]``.  This module
provides the drop-in layer that lets new strategies be added by name,
without editing the core engine or the broker:

* :class:`StrategyRegistry`  -- dict-like store mapping *name -> strategy
  class*; ``register(name, strategy_class)`` / ``get(name)`` are the API.
* :func:`load_strategy_preset` -- instantiates a named strategy class
  (with optional kwargs) so the rest of the architecture can select a
  strategy purely by configuration string.
* :func:`register` / :func:`get` -- module-level convenience wrappers over
  the *default* registry, so a strategy can be registered exactly once,
  near its own module, and picked up everywhere else.

Builtin presets are seeded into the default registry so nothing breaks out
of the box::

    >>> from signals.registry import get, load_strategy_preset
    >>> get('market_trend') is not None
    True
    >>> strat = load_strategy_preset('market_trend')
    >>> isinstance(strat, type)
    True
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from signals.ensembler import EnsemblePreset
from signals.regime import RegimeStrategy
from signals.rules import RuleStrategy
from signals.stat_arb import StatArbStrategy
from signals.strategy import MarketTrendStrategy

__all__ = [
    "StrategyRegistry",
    "register",
    "get",
    "load_strategy_preset",
]

#: Name -> strategy class that ships with the bot.
_BUILTIN_PRESETS: Dict[str, type] = {
    "market_trend": MarketTrendStrategy,
    "rule": RuleStrategy,
    "pairs_stat_arb": StatArbStrategy,
    "regime_adaptive": RegimeStrategy,
    "signal_ensemble": EnsemblePreset,
}

#: Default registry: shared by the module-level ``register``/``get``/
#: ``load_strategy_preset`` helpers.
_DEFAULT_REGISTRY: Optional[StrategyRegistry] = None


class StrategyRegistry:
    """Name -> strategy class registry.

    The registry is a plain dict with validation, so new strategies can be
    registered from anywhere in the codebase and then picked up by name by
    the engine, the CLI or tests, without touching core files.
    """

    def __init__(self) -> None:
        self._registry: Dict[str, type] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register(self, name: str, strategy_class: type) -> None:
        """Register ``strategy_class`` under ``name``.

        Parameters
        ----------
        name:
            Registry key, e.g. ``'market_trend'``.
        strategy_class:
            A class exposing ``generate(candles) -> list[Signal]``.

        Raises
        ------
        ValueError
            If ``name`` is empty.
        TypeError
            If ``strategy_class`` is not a class.
        """
        if not isinstance(name, str) or not name:
            raise ValueError(f"registry name must be a non-empty string, got {name!r}")
        if not isinstance(strategy_class, type):
            raise TypeError(f"strategy_class must be a class, got {strategy_class!r}")
        self._registry[name] = strategy_class

    def get(self, name: str) -> Optional[type]:
        """Return the strategy class registered under ``name`` (or None)."""
        return self._registry.get(name)

    def names(self) -> tuple:
        """Return the registered strategy names in registration order."""
        return tuple(self._registry)

    def has(self, name: str) -> bool:
        """Return True if a strategy is registered under ``name``."""
        return name in self._registry

    def __contains__(self, name: object) -> bool:
        """Allow ``name in registry`` membership checks."""
        return isinstance(name, str) and name in self._registry

    # ------------------------------------------------------------------
    # Bulk loading
    # ------------------------------------------------------------------

    def register_builtins(self) -> None:
        """Seed the registry with the strategies that ship with the bot."""
        self.register_all(_BUILTIN_PRESETS.items())

    def register_all(self, items: Iterable[tuple[str, type]]) -> None:
        """Register several strategies at once."""
        for name, cls in items:
            self.register(name, cls)


def _default_registry() -> StrategyRegistry:
    """Return (and lazily create) the default strategy registry."""
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = StrategyRegistry()
        _DEFAULT_REGISTRY.register_builtins()
    return _DEFAULT_REGISTRY


def register(name: str, strategy_class: type) -> None:
    """Register ``strategy_class`` under ``name`` in the *default* registry.

    Convenient for ``import signals.registry; registry.register(...)``
    style initialisation without instantiating the class explicitly.
    """
    _default_registry().register(name, strategy_class)


def get(name: str) -> Optional[type]:
    """Return the strategy class registered under ``name`` in the *default*
    registry (or None)."""
    return _default_registry().get(name)


def load_strategy_preset(name: str, **kwargs: Any) -> Any:
    """Instantiate a strategy by name (the *minimal* preset loader).

    New strategies are added by registering them (see :func:`register`) and
    then selecting them by name here -- no core file needs to be edited.

    Parameters
    ----------
    name:
        Registered strategy name, e.g. ``'market_trend'``.
    **kwargs:
        Extra arguments passed straight through to the strategy ctor.

    Returns
    -------
    Strategy instance.

    Raises
    ------
    ValueError
        If ``name`` is not registered.
    """
    strategy_class = get(name)
    if strategy_class is None:
        raise ValueError(f"unknown strategy preset: {name!r}")
    return strategy_class(**kwargs)
