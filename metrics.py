"""Structured logging to container stdout.

All events are emitted as single-line JSON records via the standard
``logging`` module.  Hugging Face Spaces captures container stdout, so
these appear in the Space's runtime logs.
"""

from __future__ import annotations

import json
import logging
import time

_log = logging.getLogger("equation_printer")
_log.setLevel(logging.INFO)
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(message)s"))
    _log.addHandler(_h)


def _emit(event: str, **extra: object) -> None:
    payload = {
        "event": event,
        "timestamp": time.time(),
        **extra,
    }
    _log.info(json.dumps(payload, default=str))


def track_page_view() -> None:
    _emit("page_view")


def track_expression(expr_text: str) -> None:
    _emit("expression", expression=expr_text)


def track_stl_download() -> None:
    _emit("stl_download")


def track_error(reason: str) -> None:
    _emit("error", reason=reason)
