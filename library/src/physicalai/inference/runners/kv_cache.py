# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""KV-cache runner for VLA policies (PI0, PI05, SmolVLA).

Phase 1 (encode): process images, language tokens, and state into a
KV cache — runs once per inference call.

Phase 2 (denoise): iterative Euler flow-matching using the cached KV
values — runs ``num_inference_steps`` times.

This runner requires two named adapters (``"encoder"`` and
``"denoise"``) and manages its own input composition, so
:class:`~physicalai.inference.model.InferenceModel` skips its normal
``_prepare_inputs`` flattening.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from physicalai.inference.runners.base import InferenceRunner

if TYPE_CHECKING:
    from physicalai.inference.adapters.base import RuntimeAdapter


def _get_output(outputs: dict[str, Any], primary: str, fallbacks: list[str]) -> np.ndarray:
    """Extract a named output with fallback keys.

    Args:
        outputs: Adapter output dict.
        primary: Preferred output key.
        fallbacks: Alternative keys to try in order.

    Returns:
        The matched output array.

    Raises:
        KeyError: If none of the keys are found.
    """
    if primary in outputs:
        return outputs[primary]
    for name in fallbacks:
        if name in outputs:
            return outputs[name]
    if len(outputs) == 1:
        return next(iter(outputs.values()))
    msg = f"Expected output '{primary}' not found. Available: {list(outputs.keys())}"
    raise KeyError(msg)


class KVCacheRunner(InferenceRunner):
    """Encode-once + iterative-denoise runner for VLA policies.

    The encoder adapter receives observations and produces a KV cache.
    The denoise adapter receives ``{x_t, timestep, prefix_pad_mask,
    past_key_*, past_value_*, ...}`` and iteratively refines a noisy
    action chunk via Euler flow-matching.

    Args:
        num_inference_steps: Number of Euler denoising iterations.
        chunk_size: Temporal horizon of the generated action chunk.
        n_action_steps: Number of action steps (may differ from chunk_size).
        action_dim: Dimensionality of the action space.
        num_layers: Number of transformer layers (informational).
        state_dim: Expected state dimension for zero-padding.
        scheduler: Scheduler type (currently only ``"euler"``).
        input_mapping: Optional mapping from observation keys to adapter
            input names (e.g. ``{"observation.state": "state"}``).
        **kwargs: Ignored — accepted for forward compatibility with
            manifest specs that may contain extra keys.
    """

    @property
    def manages_own_inputs(self) -> bool:
        """Whether this runner composes internal adapter inputs itself."""
        return True

    def __init__(
        self,
        num_inference_steps: int = 10,
        chunk_size: int = 50,
        n_action_steps: int = 50,
        action_dim: int = 6,
        num_layers: int = 18,
        state_dim: int | None = None,
        scheduler: str = "euler",
        input_mapping: dict[str, str] | None = None,
        **kwargs: Any,  # noqa: ARG002, ANN401
    ) -> None:
        """Initialize KV-cache VLA runner configuration."""
        self.num_inference_steps = num_inference_steps
        self.chunk_size = chunk_size
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.num_layers = num_layers
        self.state_dim = state_dim
        self.scheduler_type = scheduler.lower()
        self.input_mapping = input_mapping or {}

    def run(  # noqa: PLR0914
        self,
        adapter: RuntimeAdapter | dict[str, RuntimeAdapter],
        inputs: dict[str, Any],
    ) -> dict[str, Any]:
        """Run encode-once + iterative denoise and return action chunk.

        Args:
            adapter: A dict of named adapters with ``"encoder"`` and
                ``"denoise"`` keys.  A single adapter is not supported.
            inputs: Preprocessed observation inputs.

        Returns:
            Output dict containing ``"action"``.

        Raises:
            TypeError: If *adapter* is not a dict of adapters.
        """
        if not isinstance(adapter, dict):
            msg = "KVCacheRunner requires a dict of adapters with 'encoder' and 'denoise' keys"
            raise TypeError(msg)

        encoder_adapter = adapter["encoder"]
        denoise_adapter = adapter["denoise"]

        obs = dict(inputs)

        # Apply optional input-name mapping
        if self.input_mapping:
            mapped: dict[str, Any] = {}
            for obs_key, value in obs.items():
                onnx_key = self.input_mapping.get(obs_key, obs_key)
                mapped[onnx_key] = value
            obs = mapped

        # Ensure float32 for all array inputs
        for key in list(obs):
            if isinstance(obs[key], np.ndarray):
                obs[key] = obs[key].astype(np.float32)

        first_val = next(iter(obs.values()))
        batch_size = first_val.shape[0] if first_val.ndim > 1 else 1

        # Auto-add image masks if missing
        num_images = sum(1 for k in obs if k.startswith("image_"))
        for i in range(num_images):
            mask_key = f"img_mask_{i}"
            if mask_key not in obs:
                obs[mask_key] = np.ones((batch_size,), dtype=np.float32)

        # Pad state to expected dimension if needed
        if self.state_dim is not None and "state" in obs:
            state = obs["state"]
            current_dim = state.shape[-1]
            if current_dim < self.state_dim:
                padding = np.zeros((*state.shape[:-1], self.state_dim - current_dim), dtype=state.dtype)
                obs["state"] = np.concatenate([state, padding], axis=-1)

        # Phase 1: Encode
        encoder_outputs = encoder_adapter.predict(obs)
        prefix_pad_mask = encoder_outputs.get("prefix_pad_mask")
        if prefix_pad_mask is None:
            prefix_pad_mask = next(iter(encoder_outputs.values()))

        kv_cache = {k: v for k, v in encoder_outputs.items() if k.startswith("past_")}

        # Phase 2: Iterative denoise (Euler flow-matching)
        action_shape = (batch_size, self.chunk_size, self.action_dim)
        x_t = np.random.default_rng().standard_normal(action_shape).astype(np.float32)

        dt = -1.0 / self.num_inference_steps

        for step in range(self.num_inference_steps):
            t = 1.0 + step * dt
            timestep = np.full((batch_size,), t, dtype=np.float32)

            denoise_inputs: dict[str, Any] = {
                "x_t": x_t,
                "timestep": timestep,
                "prefix_pad_mask": prefix_pad_mask,
                **kv_cache,
            }

            if "state" in obs:
                denoise_inputs["state"] = obs["state"]

            outputs = denoise_adapter.predict(denoise_inputs)
            v_t = _get_output(outputs, "v_t", ["velocity"])
            x_t += dt * v_t

        return {"action": x_t}

    def reset(self) -> None:
        """No-op — KV-cache runner is stateless between episodes."""

    def __repr__(self) -> str:
        """Return string representation of the runner."""
        return (
            f"{self.__class__.__name__}("
            f"steps={self.num_inference_steps}, "
            f"chunk_size={self.chunk_size}, "
            f"action_dim={self.action_dim})"
        )
