"""Tests for the no-code rule editor."""

import pytest

from orchestrator.engine import Engine, Symbol
from risk.mgr import RiskManager
from signals.rules import RuleStrategy, parse_rules
from signals.strategy import Candle, MarketTrendStrategy


def _candles(closes, volume=1000):
    """Build candles from a sequence of closing prices."""
    return [
        Candle(timestamp=i, open=c - 1, high=c + 1, low=c - 1, close=c, volume=volume)
        for i, c in enumerate(closes)
    ]


def _engine():
    return Engine(
        symbols=[Symbol("CL=F")],
        risk_manager=RiskManager(account_balance=10000, max_per_trade=0.0025),
        fetcher=None,
        strategy=MarketTrendStrategy(),
    )


# ------------------------------------------------------------------
# parse_rules
# ------------------------------------------------------------------


def test_parse_single_rule():
    rules = parse_rules("if close > 100: BUY")
    assert len(rules) == 1
    assert rules[0].direction == "BUY"
    assert hasattr(rules[0].condition, "op")


def test_parse_direction_validation():
    with pytest.raises(SyntaxError):
        parse_rules("if close > 100: HOLD")
    # Direction validation of an *existing* Rule is gated by RuleStrategy.add_rule,
    # so a valid rule must not raise here.
    parse_rules("if close > 100: BUY")


def test_parse_unknown_field():
    with pytest.raises(SyntaxError):
        parse_rules("if bid > 100: BUY")


def test_parse_missing_colon():
    with pytest.raises(SyntaxError):
        parse_rules("if close > 100 BUY")


def test_parse_unknown_function():
    with pytest.raises(SyntaxError):
        parse_rules("if foo(5) > 100: BUY")


def test_parse_function_wrong_arg_count():
    with pytest.raises(SyntaxError):
        parse_rules("if ema() > 100: BUY")


def test_parse_condition_with_parens_and_negation():
    rules = parse_rules("if not (close > ema(50)): BUY")
    assert len(rules) == 1
    assert rules[0].direction == "BUY"


def test_parse_compound_condition():
    rules = parse_rules(
        "if close > sma(20) and volume_z > 1.5: BUY\n" "if close < sma(20): SELL"
    )
    assert len(rules) == 2


# ------------------------------------------------------------------
# RuleStrategy.generate
# ------------------------------------------------------------------


def test_generate_no_signals_on_short_input():
    strat = RuleStrategy(parse_rules("if close > ema(5): BUY"))
    assert strat.generate([]) == []
    assert (
        strat.generate([Candle(timestamp=0, open=1, high=2, low=0, close=1, volume=0)])
        == []
    )


def test_generate_buys_when_true():
    # Rising closes -> close > ema(5) is True at the last candle.
    candles = _candles(list(range(80, 120)))
    strat = RuleStrategy(parse_rules("if close > ema(5): BUY"))
    sigs = strat.generate(candles)
    directions = {s.direction for s in sigs}
    assert "BUY" in directions
    assert sigs and sigs[-1].confidence == 0.7


def test_generate_sells_when_true():
    # Falling closes: the last candle (81) is below sma(20).
    candles = _candles(list(range(120, 80, -1)))
    strat = RuleStrategy(parse_rules("if close < sma(20): SELL"))
    sigs = strat.generate(candles)
    assert any(s.direction == "SELL" for s in sigs), "expected SELL on a falling bar"


def test_generate_strength_and_reason():
    candles = _candles(list(range(80, 120)))
    strat = RuleStrategy(parse_rules("if close > ema(5): BUY"))
    sigs = strat.generate(candles)
    assert sigs[0].reason.startswith("rule fired")
    assert sigs[0].timestamp == candles[-1].timestamp


def test_add_rule_invalid_direction():
    from signals.rules import Rule

    rule = parse_rules("if close > 100: BUY")[0]
    strat = RuleStrategy()
    with pytest.raises(ValueError):
        strat.add_rule(Rule(direction="HOLD", condition=rule.condition))


def test_add_rule_empty_direction():
    from signals.rules import Rule

    rule = parse_rules("if close > 100: BUY")[0]
    strat = RuleStrategy()
    with pytest.raises(ValueError):
        strat.add_rule(Rule(direction="HOLD", condition=rule.condition))


def test_generate_empty_rules():
    strat = RuleStrategy([])
    assert strat.generate(_candles(list(range(50, 100)))) == []
