# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Iterative runner for diffusion/flow-matching policies.

Supports Euler (flow-matching), DDPM, and DDIM schedulers.
The scheduler is configured from runner parameters in the manifest.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import InferenceRunner

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from physicalai.inference.adapters.base import RuntimeAdapter


def _get_output(outputs: dict[str, Any], primary: str, fallbacks: list[str]) -> np.ndarray:
    if primary in outputs:
        return outputs[primary]
    for name in fallbacks:
        if name in outputs:
            return outputs[name]
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    msg = f"Expected output '{primary}' not found. Available: {list(outputs.keys())}"
    raise KeyError(msg)


class IterativeRunner(InferenceRunner):
    """Multi-step iterative denoising (diffusion / flow-matching)."""

    @property
    def manages_own_inputs(self) -> bool:
        """Whether this runner composes internal adapter inputs itself."""
        return True

    def __init__(
        self,
        num_inference_steps: int = 10,
        scheduler: str = "euler",
        action_dim: int = 6,
        chunk_size: int = 16,
        horizon: int | None = None,
        **scheduler_kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize iterative denoising runner configuration."""
        self.num_inference_steps = num_inference_steps
        self.scheduler_type = scheduler.lower()
        self.action_dim = action_dim
        self.chunk_size = horizon or chunk_size
        self._scheduler_kwargs = scheduler_kwargs
        self._diffusion_scheduler = self._create_scheduler()

    def run(self, adapters: dict[str, RuntimeAdapter], inputs: dict[str, Any]) -> dict[str, Any]:
        """Run iterative denoising and return an action chunk.

        Returns:
            Output dict containing ``"action"``.
        """
        adapter = adapters.get("model")
        if adapter is None:
            adapter = next(iter(adapters.values()))

        first_val = next(iter(inputs.values()))
        batch_size = first_val.shape[0] if first_val.ndim > 1 else 1
        action_shape = (batch_size, self.chunk_size, self.action_dim)
        x_t = np.random.default_rng().standard_normal(action_shape).astype(np.float32)

        if self._diffusion_scheduler is not None:
            x_t = self._run_diffusion_loop(adapter, x_t, inputs, batch_size)
        else:
            x_t = self._run_euler_loop(adapter, x_t, inputs, batch_size)

        return {"action": x_t}

    def _run_euler_loop(
        self,
        adapter: RuntimeAdapter,
        x_t: NDArray[np.floating],
        obs: dict[str, Any],
        batch_size: int,
    ) -> NDArray[np.floating]:
        dt = -1.0 / self.num_inference_steps

        for step in range(self.num_inference_steps):
            t = 1.0 + step * dt
            timestep = np.full((batch_size,), t, dtype=np.float32)
            step_inputs = {"x_t": x_t, "timestep": timestep, **obs}
            outputs = adapter.predict(step_inputs)
            v_t = _get_output(outputs, "v_t", ["velocity"])
            x_t += dt * v_t

        return x_t

    def _run_diffusion_loop(
        self,
        adapter: RuntimeAdapter,
        x_t: NDArray[np.floating],
        obs: dict[str, Any],
        batch_size: int,  # noqa: ARG002
    ) -> NDArray[np.floating]:
        scheduler = self._diffusion_scheduler
        timesteps = scheduler.set_timesteps(self.num_inference_steps)

        for t in timesteps:
            timestep_arr = np.array([t], dtype=np.float32)
            step_inputs = {"x_t": x_t, "timestep": timestep_arr, **obs}
            outputs = adapter.predict(step_inputs)
            model_output = _get_output(outputs, "v_t", ["velocity", "noise_pred"])
            x_t = scheduler.step(model_output, int(t), x_t)

        return x_t

    def _create_scheduler(self) -> object:
        if self.scheduler_type == "euler":
            return None
        if self.scheduler_type == "ddpm":
            from physicalai.inference.runners._schedulers import DDPMScheduler  # noqa: PLC0415, PLC2701

            return DDPMScheduler(self._scheduler_kwargs)
        if self.scheduler_type == "ddim":
            from physicalai.inference.runners._schedulers import DDIMScheduler  # noqa: PLC0415, PLC2701

            return DDIMScheduler(self._scheduler_kwargs)
        msg = f"Unknown scheduler: {self.scheduler_type!r}. Supported: euler, ddpm, ddim"
        raise ValueError(msg)
