"""Natural-language math expression parser.

Converts free-form text like ``"sinx cosy"``, ``"z = x^2 + y^2"``, or
``"plot sin(x)*cos(y)"`` into a valid ``sympy.Expr`` involving only the
symbols ``x`` and ``y``.
"""

from __future__ import annotations

import re

import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    split_symbols,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Math function names, sorted longest-first so greedy regex matching works
# correctly (e.g. "arcsin" matched before "sin").
_FUNCTION_NAMES = sorted(
    [
        "sin",
        "cos",
        "tan",
        "cot",
        "sec",
        "csc",
        "arcsin",
        "arccos",
        "arctan",
        "sinh",
        "cosh",
        "tanh",
        "exp",
        "log",
        "ln",
        "sqrt",
        "abs",
        "sign",
        "erf",
        "gamma",
        "floor",
        "ceil",
    ],
    key=len,
    reverse=True,
)

# Compiled regex that matches a function name immediately followed by a
# single letter  —  e.g. "sinx" → ("sin", "x"), "cosy" → ("cos", "y").
_FUNC_CALL_RE = re.compile(
    r"(?<!\w)(" + "|".join(re.escape(f) for f in _FUNCTION_NAMES) + r")([a-z])(?!\w)"
)

# Stripped from the *beginning* of the input (case-insensitive).
_PREFIX_PATTERNS = [
    re.compile(r"^\s*(?:z|f\s*\(\s*x\s*,\s*y\s*\))\s*[:=]\s*", re.IGNORECASE),
    re.compile(r"^\s*(?:plot|graph|draw|show)\s+", re.IGNORECASE),
]

# SymPy parser transformations: standard set + split_symbols for "xy" → "x*y"
# + implicit_multiplication_application for "2x" → "2*x" and "x y" → "x*y".
_TRANSFORMATIONS = standard_transformations + (
    split_symbols,
    implicit_multiplication_application,
)

# Cache the symbols so we don't recreate them on every parse.
_X, _Y = sympy.symbols("x y")

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def preprocess(text: str) -> str:
    """Clean up raw user input before SymPy sees it.

    Steps (in order):

    1. Strip leading prefixes like ``z =``, ``plot``, ``f(x,y) =``.
    2. Replace ``^`` with ``**`` (Python exponentiation syntax).
    3. Expand shorthand function calls: ``sinx`` → ``sin(x)``,
       ``cos y`` → ``cos(y)`` (when ``y`` is a bare letter after the
       function name).
    4. Insert ``*`` for implicit multiplication between a closing paren
       and a letter, and between a digit and a letter.
    """
    original = text

    # 1. Strip prefixes
    for pat in _PREFIX_PATTERNS:
        text = pat.sub("", text)

    # 2. Caret → double-star
    text = text.replace("^", "**")

    # 3. Expand "funcname letter" → "funcname(letter)"
    text = _FUNC_CALL_RE.sub(r"\1(\2)", text)

    # 4a. Insert * between ")" and a letter  —  sin(x)cos(y) → sin(x)*cos(y)
    text = re.sub(r"\)\s*([a-zA-Z])", r")*\1", text)

    # 4b. Insert * between a digit and a letter  —  2x → 2*x
    text = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", text)

    # 4c. Insert * between a digit and '('  —  15(x+y) → 15*(x+y)
    text = re.sub(r"(\d)\(", r"\1*(", text)

    # 5. Lowercase 'e' as Euler's number when used as base of exponentiation
    #    e^... → E^...  (later caret → **, and SymPy knows E = exp(1))
    text = re.sub(r"(?<!\w)e\s*\*\s*\*", "E**", text)

    return text.strip()


def parse_expression(text: str) -> tuple[sympy.Expr | None, str | None]:
    """Parse a natural-language math expression into a SymPy expression.

    Returns:
        ``(expr, None)`` on success, ``(None, error_message)`` on failure.
        The error message is suitable for display to the user.
    """
    raw = text.strip()
    if not raw:
        return None, "Please enter an expression."

    cleaned = preprocess(raw)
    if not cleaned:
        return None, (
            "Could not extract an expression from your input. "
            "Try something like ``sin(x)*cos(y)`` or ``x^2 + y^2``."
        )

    local_dict = {"x": _X, "y": _Y}

    # --- Pass 1: fast sympify for well-formed input ---
    try:
        expr = sympy.sympify(cleaned, locals=local_dict)
    except sympy.SympifyError:
        expr = None

    # --- Pass 2: richer parser with implicit multiplication ---
    if expr is None:
        try:
            expr = parse_expr(
                cleaned,
                local_dict=local_dict,
                transformations=_TRANSFORMATIONS,
            )
        except Exception as exc:
            return None, _build_error(str(exc), raw)

    return _validate(expr, raw)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _validate(
    expr: sympy.Expr, original: str
) -> tuple[sympy.Expr | None, str | None]:
    """Ensure the expression only uses the symbols ``x`` and ``y``."""
    allowed = {_X, _Y}
    free = expr.free_symbols
    unknown = free - allowed
    if unknown:
        names = ", ".join(sorted(str(s) for s in unknown))
        return None, (
            f"Unknown variable{'s' if len(unknown) > 1 else ''}: **{names}**. "
            f"Please use only **x** and **y** as variables."
        )
    return expr, None


def _build_error(detail: str, original: str) -> str:
    """Turn a raw SymPy / parse error into a short, user-friendly message."""
    # Try to give a helpful hint for the most common mistakes.
    low = detail.lower()
    if "token" in low or "syntax" in low:
        return (
            f"Could not understand the expression: ``{original}``.\n\n"
            "Check for missing operators (e.g. write ``sin(x)*cos(y)`` "
            "instead of ``sin(x)cos(y)``) or unmatched parentheses."
        )
    return f"Could not parse the expression: ``{original}``.\n\n{detail}"
