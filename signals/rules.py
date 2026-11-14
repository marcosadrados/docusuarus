"""No-code rule editor for the scalping trading bot.

Lets users author scalping rules in plain language and have them compiled
into a full ``generate(candles) -> list[Signal]`` strategy.  Rules are
registered as presets in the :class:`signals.registry.StrategyRegistry`, so
they can be selected by name in :attr:`orchestrator.main.Orchestrator` and
installed via the future presets marketplace (see :mod:`monitoring.notifier`
for the signal layer).

DSL (one rule per line)::

    if close > ema(5) and volume > 1000000: BUY
    if close < ema(20): SELL
    if rsi(14) < 30: BUY
    if rsi(14) > 70: SELL
    if volume_z > 1.5 and close < sma(20): SELL
    if not (close > ema(50)): BUY

Supported fields: ``close``, ``open``, ``high``, ``low``, ``volume``.
Supported indicators: ``ema(N)``, ``sma(N)``, ``rsi(N)``, ``volume_z``.
Operators: ``> < >= <= == !=``, and/or/not, parentheses, unary minus.

Indicator periods available: 5, 10, 14, 20, 50.

Example
-------
>>> from signals.rules import parse_rules, RuleStrategy
>>> rules = parse_rules("if close > ema(5): BUY")
>>> strat = RuleStrategy(rules)
>>> sigs = strat.generate(candles)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from signals.strategy import Signal

__all__ = [
    "RuleStrategy",
    "parse_rules",
]

#: Fields a rule may reference (shared by parser & evaluator).
ALLOWED_FIELDS = ("close", "open", "high", "low", "volume")

#: Periods for which indicators are pre-computed.
INDICATOR_PERIODS = (5, 10, 14, 20, 50)

#: Regex: split on ':', which terminates the condition and begins the direction.
_RULE_SPLIT = re.compile(r"\s*:\s*")

#: Tokenizer operators, longest-first so '>=' matches before '>'.
_TOKEN_RE = re.compile(
    r"\s*(>=|<=|==|!=|<|>|&&||\|||[-+*/%()]|\b(?:and|or|not)\b|[A-Za-z_][A-Za-z0-9_]*"
    r"|[0-9]+(?:\.[0-9]+)?)\s*",
    re.MULTILINE,
)

_COMPARE_OPS = (">", "<", ">=", "<=", "==", "!=")


@dataclass
class Rule:
    """A single rule: one direction with one (possibly compound) condition."""

    direction: str  # BUY or SELL
    condition: Any  # AST node from the parser

    def __post_init__(self) -> None:
        direction = self.direction.upper()
        if direction not in ("BUY", "SELL"):
            raise ValueError(
                f"rule direction must be BUY or SELL, got {self.direction!r}"
            )
        self.direction = direction


# ------------------------------------------------------------------
# Expression AST
# ------------------------------------------------------------------


@dataclass
class Expr:
    """Base class for a parsed condition expression."""


@dataclass
class Lit(Expr):
    value: float


@dataclass
class Var(Expr):
    name: str


@dataclass
class BinOp(Expr):
    op: str
    left: Expr
    right: Expr


@dataclass
class Not(Expr):
    expr: Expr


@dataclass
class Paren(Expr):
    expr: Expr


@dataclass
class _Call(Expr):
    name: str
    arg: Expr


# ------------------------------------------------------------------
# Tokenizer
# ------------------------------------------------------------------


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for match in _TOKEN_RE.finditer(text):
        tok = match.group(1)
        if tok:
            tokens.append(tok)
    return tokens


# ------------------------------------------------------------------
# Recursive-descent parser (intra-expression)
# ------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: List[str]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[str]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def advance(self) -> Optional[str]:
        tok = self.peek()
        if tok is not None:
            self.pos += 1
        return tok

    def _advance_expect(self) -> str:
        """Consume and return the next token; never returns None."""
        tok = self.advance()
        if tok is None:
            raise SyntaxError("unexpected end of expression")
        return tok

    def parse(self) -> Expr:
        expr = self._or_expr()
        if self.peek() is not None:
            raise SyntaxError(f"unexpected token near {self.peek()!r}")
        return expr

    # or -> and -> not -> comparison -> additive -> multiplicative -> unary -> atom
    def _or_expr(self) -> Expr:
        left = self._and_expr()
        while self.peek() == "or":
            self.advance()
            right = self._and_expr()
            left = BinOp("or", left, right)
        return left

    def _and_expr(self) -> Expr:
        left = self._not_expr()
        while self.peek() == "and":
            self.advance()
            right = self._not_expr()
            left = BinOp("and", left, right)
        return left

    def _not_expr(self) -> Expr:
        if self.peek() == "not":
            self.advance()
            return Not(self._not_expr())
        return self._comparison()

    def _comparison(self) -> Expr:
        left = self._additive()
        if self.peek() not in _COMPARE_OPS:
            return left
        op = self._advance_expect()
        right = self._additive()
        return BinOp(op, left, right)

    def _additive(self) -> Expr:
        left = self._multiplicative()
        while self.peek() in ("+", "-"):
            op = self._advance_expect()
            right = self._multiplicative()
            left = BinOp(op, left, right)
        return left

    def _multiplicative(self) -> Expr:
        left = self._unary()
        while self.peek() in ("*", "/", "%"):
            op = self._advance_expect()
            right = self._unary()
            left = BinOp(op, left, right)
        return left

    def _unary(self) -> Expr:
        if self.peek() == "-":
            self.advance()
            return BinOp("-", Lit(0.0), self._unary())
        return self._atom()

    def _atom(self) -> Expr:
        tok = self.advance()
        if tok is None:
            raise SyntaxError("unexpected end of expression")
        if tok == "(":
            inner = self._or_expr()
            if self.advance() != ")":
                raise SyntaxError("missing closing paren")
            return Paren(inner)
        if tok in (
            "and",
            "or",
            "not",
            ">",
            "<",
            ">=",
            "<=",
            "==",
            "!=",
            "+",
            "-",
            "*",
            "/",
            "%",
        ):
            raise SyntaxError(f"unexpected token {tok!r} near {tok!r}")
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tok):
            if self.peek() == "(":
                self.advance()  # consume '('
                args = []
                if self.peek() != ")":
                    args.append(self._or_expr())
                    while self.peek() == ",":
                        self.advance()
                        args.append(self._or_expr())
                if self.advance() != ")":
                    raise SyntaxError("missing closing paren")
                if len(args) != 1:
                    raise SyntaxError(f"{tok}() takes exactly 1 argument")
                return _Call(name=tok, arg=args[0])
            return Var(tok)
        # Number
        try:
            return Lit(float(tok))
        except ValueError:
            raise SyntaxError(f"unexpected token {tok!r}") from None


# ------------------------------------------------------------------
# Indicators (computed per generate() call)
# ------------------------------------------------------------------


def _closes(candles: List[Any]) -> List[float]:
    return [float(c.close) for c in candles]


def _volume(candles: List[Any]) -> List[float]:
    return [float(c.volume) for c in candles]


def _ema(values: List[float], period: int) -> List[float]:
    n = len(values)
    out = [0.0] * n
    if n < period:
        return out
    mult = 2 / (period + 1)
    e = sum(values[:period]) / period
    out[period - 1] = e
    for i in range(period, n):
        e = (values[i] - e) * mult + e
        out[i] = e
    return out


def _sma(values: List[float], period: int) -> List[float]:
    n = len(values)
    out = [0.0] * n
    if n < period:
        return out
    window = values[:period]
    out[period - 1] = sum(window) / period
    for i in range(period, n):
        window = window[1:] + [values[i]]
        out[i] = sum(window) / period
    return out


def _rsi(values: List[float], period: int = 14) -> List[float]:
    n = len(values)
    out = [0.0] * n
    if n < period + 1:
        return out
    changes = [values[i] - values[i - 1] for i in range(1, n)]
    gains = [max(c, 0.0) for c in changes]
    losses = [max(-c, 0.0) for c in changes]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period
    out[period] = 50.0  # neutral at the first valid index
    for i in range(period, n - 1):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        if avg_loss == 0:
            out[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            out[i + 1] = 100 - 100 / (1 + rs)
    return out


def _volume_z(values: List[float]) -> List[float]:
    n = len(values)
    if n == 0:
        return []
    mean = sum(values) / n
    if mean == 0:
        return [0.0] * n
    var = sum((v - mean) ** 2 for v in values) / n
    std = var**0.5
    if std == 0:
        return [0.0] * n
    return [(v - mean) / std for v in values]


def _compute_indicators(candles: List[Any]) -> Dict[str, List[float]]:
    closes = _closes(candles)
    volumes = _volume(candles)
    out: Dict[str, List[float]] = {"volume_z": _volume_z(volumes)}
    for period in INDICATOR_PERIODS:
        out[f"ema_{period}"] = _ema(closes, period)
        out[f"sma_{period}"] = _sma(closes, period)
        out[f"rsi_{period}"] = _rsi(closes, period)
    return out


# ------------------------------------------------------------------
# Rule parser
# ------------------------------------------------------------------


def parse_rules(dsl_text: str) -> List[Rule]:
    """Parse a multiline rule DSL into a list of Rule objects."""
    rules: List[Rule] = []
    for lineno, raw in enumerate(dsl_text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if not line.lower().startswith("if "):
            raise SyntaxError(
                f"line {lineno}: expected 'if <condition>: <BUY|SELL>', got {line!r}"
            )
        body = line[3:]
        parts = _RULE_SPLIT.split(body, 1)
        if len(parts) != 2:
            raise SyntaxError(
                f"line {lineno}: missing ':' separator between condition and direction"
            )
        left, right = parts
        cond_tokens = _tokenize(left)
        right = right.strip().upper()
        if right not in ("BUY", "SELL"):
            raise SyntaxError(
                f"line {lineno}: direction must be BUY or SELL, got {right!r}"
            )
        parser = _Parser(cond_tokens)
        condition = parser.parse()
        _validate_vars(condition)
        rules.append(Rule(direction=right, condition=condition))
    return rules


def _is_indicator(name: str) -> bool:
    """True if a Var node names a pre-computed indicator (not a raw field)."""
    return name == "volume_z" or re.match(r"^(ema|sma|rsi)_\d+$", name) is not None


def _validate_vars(expr: Expr) -> None:
    def _walk(node: Expr, path: List[str]) -> None:
        if isinstance(node, Var):
            if node.name not in ALLOWED_FIELDS and not _is_indicator(node.name):
                raise SyntaxError(
                    f"line {':'.join(path + [node.name])}: unknown field "
                    f"{node.name!r}"
                )
        elif isinstance(node, BinOp):
            _walk(node.left, path + ["left"])
            _walk(node.right, path + ["right"])
        elif isinstance(node, Not):
            _walk(node.expr, path + ["not"])
        elif isinstance(node, Paren):
            _walk(node.expr, path + ["paren"])
        elif isinstance(node, _Call):
            if node.name not in ("ema", "sma", "rsi", "volume_z"):
                raise SyntaxError(
                    f"line {':'.join(path)}: unknown function {node.name!r}"
                )
            if node.name != "volume_z" and not isinstance(node.arg, Lit):
                raise SyntaxError(
                    f"line {':'.join(path)}: {node.name}() expects a numeric period"
                )
        elif isinstance(node, Lit):
            pass
        else:
            raise SyntaxError(f"unexpected expression node {type(node).__name__!r}")

    _walk(expr, [])


# ------------------------------------------------------------------
# Compiler: evaluate an AST against candles + indicators
# ------------------------------------------------------------------


class _Evaluator:
    """Returns a list of floats per candle; truthy = non-zero."""

    def __init__(self, candles: List[Any], indicators: Dict[str, List[float]]) -> None:
        self.candles = candles
        self.indicators = indicators

    def evaluate(self, expr: Expr) -> List[float]:
        out = []
        for i in range(len(self.candles)):
            out.append(self._eval(expr, i))
        return out

    def _eval(self, expr: Expr, i: int) -> float:
        if isinstance(expr, Lit):
            return expr.value
        if isinstance(expr, Var):
            if expr.name in self.indicators:
                return float(self.indicators[expr.name][i])
            return float(getattr(self.candles[i], expr.name, 0.0))
        if isinstance(expr, BinOp):
            left = self._eval(expr.left, i)
            right = self._eval(expr.right, i)
            op = expr.op
            if op in (">", "<", ">=", "<=", "==", "!="):
                return 1.0 if self._cmp(left, right, op) else 0.0
            if op == "and":
                return 1.0 if (left != 0.0 and right != 0.0) else 0.0
            if op == "or":
                return 1.0 if (left != 0.0 or right != 0.0) else 0.0
            if op == "-":
                return left - right
            if op == "+":
                return left + right
            if op == "*":
                return left * right
            if op == "/":
                return left / right if right != 0.0 else 0.0
            if op == "%":
                return left % right if right != 0.0 else 0.0
            raise ValueError(f"unknown operator {op!r}")
        if isinstance(expr, Not):
            return 1.0 if self._eval(expr.expr, i) == 0.0 else 0.0
        if isinstance(expr, Paren):
            return self._eval(expr.expr, i)
        if isinstance(expr, _Call):
            return self._eval_call(expr, i)
        return 0.0

    @staticmethod
    def _cmp(a: float, b: float, op: str) -> bool:
        if op == ">":
            return a > b
        if op == "<":
            return a < b
        if op == ">=":
            return a >= b
        if op == "<=":
            return a <= b
        if op == "==":
            return a == b
        if op == "!=":
            return a != b
        return False

    def _eval_call(self, call: _Call, i: int) -> float:
        if call.name == "volume_z":
            return float(self.indicators["volume_z"][i])
        period = int(self._eval(call.arg, i))
        key = f"{call.name}_{period}"
        if key not in self.indicators:
            return 0.0
        return float(self.indicators[key][i])


# ------------------------------------------------------------------
# RuleStrategy
# ------------------------------------------------------------------


class RuleStrategy:
    """A strategy built from compiled Rule objects."""

    def __init__(self, rules: Optional[List[Rule]] = None) -> None:
        self.rules = list(rules) if rules is not None else []

    def add_rule(self, rule: Rule) -> None:
        direction = rule.direction.upper()
        if direction not in ("BUY", "SELL"):
            raise ValueError("rule direction must be BUY or SELL")
        self.rules.append(Rule(direction=direction, condition=rule.condition))

    def generate(self, candles: List[Any]) -> List[Signal]:
        if len(candles) < 2:
            return []
        indicators = _compute_indicators(candles)
        ev = _Evaluator(candles, indicators)
        signals: List[Signal] = []
        for rule in self.rules:
            values = ev.evaluate(rule.condition)
            if values and values[-1] != 0.0:
                signals.append(
                    Signal(
                        direction=rule.direction,
                        confidence=0.7,
                        strength=0.5,
                        reason=f"rule fired: {rule.condition!r}",
                        timestamp=candles[-1].timestamp,
                    )
                )
        return signals

    def __len__(self) -> int:
        return len(self.rules)
