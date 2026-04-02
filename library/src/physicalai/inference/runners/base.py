# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base inference runner interface.

Runners encapsulate *how* inference runs — the execution procedure —
separate from *what* runs (the adapter/backend) and *where* (the device).

A runner is a composable unit: runners can wrap other runners to add
behavior (e.g. action chunking wraps any base runner to add temporal
buffering — the GoF Decorator pattern).

Runners receive a **dict of adapters** keyed by role name so that
multi-artifact policies (e.g. VLA with ``encoder`` + ``denoise``) can
address each sub-model independently.  Simple policies use
``{"model": adapter}``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter


class InferenceRunner(ABC):
    """Abstract base class for inference runners.

    A runner defines *how* a single call to the inference model
    translates into one or more adapter calls and how outputs are
    post-processed.

    Runners are composable: a runner can wrap another runner to add
    behavior without modifying the inner runner's logic. For example,
    ``ActionChunking(SinglePass())`` adds temporal buffering around
    a single forward pass.

    Subclasses must implement:
    - ``run`` — execute inference given adapters and prepared inputs
    - ``reset`` — clear any internal state between episodes

    Examples:
        >>> runner = SinglePass()
        >>> outputs = runner.run({"model": adapter}, inputs)
        >>> action = outputs["action"]
        >>> runner.reset()  # new episode
    """

    @abstractmethod
    def run(
        self,
        adapters: dict[str, RuntimeAdapter],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute one inference step and return model outputs.

        Args:
            adapters: Named runtime adapters.  Simple policies pass
                ``{"model": adapter}``; multi-artifact policies pass
                one entry per sub-model (e.g. ``{"encoder": …,
                "denoise": …}``).
            inputs: Pre-processed model inputs (flat dict of numpy arrays).

        Returns:
            Dict mapping output names to numpy arrays.
        """

    @property
    def manages_own_inputs(self) -> bool:
        """Whether this runner composes its own adapter inputs.

        Runners that generate internal tensors (e.g. ``x_t``, ``timestep``
        for diffusion) should return ``True`` so that the model skips
        input filtering against adapter input names.

        Default is ``False`` — the model may filter/flatten user inputs
        to match the adapter's expected signature.
        """
        return False

    def reset(self) -> None:  # noqa: B027
        """Reset internal state for a new episode.

        Runners that maintain state between calls (e.g. action queues)
        must clear it here. Stateless runners can no-op.
        """

    def __repr__(self) -> str:
        """Return string representation of the runner."""
        return f"{self.__class__.__name__}()"
