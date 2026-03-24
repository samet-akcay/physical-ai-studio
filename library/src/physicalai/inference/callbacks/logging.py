# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Logging callback for inference observability.

Emits structured log messages at each inference lifecycle event using
the stdlib :mod:`logging` module.  Useful for production observability
and debugging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, override

from physicalai.inference.callbacks.base import Callback

if TYPE_CHECKING:
    from physicalai.inference.model import InferenceModel

_logger = logging.getLogger("physicalai.inference")


class LoggingCallback(Callback):
    """Log inference events via stdlib logging.

    Args:
        level: Log level for prediction events (default ``DEBUG``).
        logger: Logger instance to use.  Defaults to
            ``physicalai.inference``.

    Examples:
        >>> import logging
        >>> logging.basicConfig(level=logging.DEBUG)
        >>> model = InferenceModel.load("./exports/act", callbacks=[LoggingCallback()])
        >>> model.select_action(obs)  # logs predict_start / predict_end
    """

    def __init__(
        self,
        level: int = logging.DEBUG,
        logger: logging.Logger | None = None,
    ) -> None:
        """Initialise with log *level* and optional *logger*."""
        self._level = level
        self._logger = logger or _logger

    @override
    def on_predict_start(self, inputs: dict[str, Any]) -> None:
        """Log prediction start with input keys."""
        self._logger.log(self._level, "predict_start | keys=%s", list(inputs.keys()))

    @override
    def on_predict_end(self, outputs: dict[str, Any]) -> None:
        """Log prediction end with output keys."""
        self._logger.log(self._level, "predict_end | keys=%s", list(outputs.keys()))

    @override
    def on_reset(self) -> None:
        """Log model reset."""
        self._logger.log(self._level, "reset")

    @override
    def on_load(self, model: InferenceModel) -> None:
        """Log model load completion."""
        self._logger.log(self._level, "model_loaded | %r", model)

    @override
    def __repr__(self) -> str:
        """Return string representation of the logging callback."""
        return f"LoggingCallback(level={logging.getLevelName(self._level)})"
