# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Pure-numpy diffusion schedulers (DDPM, DDIM) for iterative runner.

Mirrors the lerobot export schedulers — no PyTorch dependency at runtime.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray  # noqa: TC002


def _compute_betas(
    num_train_timesteps: int,
    beta_start: float,
    beta_end: float,
    beta_schedule: str,
) -> NDArray[np.floating]:
    """Compute beta schedule for diffusion process.

    Args:
        num_train_timesteps: Total number of training timesteps.
        beta_start: Starting beta value.
        beta_end: Ending beta value.
        beta_schedule: Schedule type (``"linear"``, ``"scaled_linear"``,
            or ``"squaredcos_cap_v2"``).

    Returns:
        Array of beta values for each timestep.

    Raises:
        ValueError: If beta_schedule is not recognized.
    """
    if beta_schedule == "linear":
        return np.linspace(beta_start, beta_end, num_train_timesteps, dtype=np.float32)
    if beta_schedule == "scaled_linear":
        return np.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps, dtype=np.float32) ** 2
    if beta_schedule == "squaredcos_cap_v2":
        return _betas_for_alpha_bar(num_train_timesteps)
    msg = f"Unknown beta_schedule: {beta_schedule}"
    raise ValueError(msg)


def _betas_for_alpha_bar(num_train_timesteps: int, max_beta: float = 0.999) -> NDArray[np.floating]:
    """Compute betas from cosine alpha-bar schedule.

    Args:
        num_train_timesteps: Total number of training timesteps.
        max_beta: Maximum allowed beta value.

    Returns:
        Array of beta values.
    """

    def alpha_bar(t: float) -> float:
        return np.cos((t + 0.008) / 1.008 * np.pi / 2) ** 2

    betas = []
    for i in range(num_train_timesteps):
        t1 = i / num_train_timesteps
        t2 = (i + 1) / num_train_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas, dtype=np.float32)


class DDPMScheduler:
    """Denoising Diffusion Probabilistic Model scheduler (pure numpy).

    Args:
        config: Scheduler configuration dict with keys like
            ``num_train_timesteps``, ``prediction_type``, ``beta_schedule``, etc.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.num_train_timesteps: int = config.get("num_train_timesteps", 1000)
        self.prediction_type: str = config.get("prediction_type", "epsilon")
        self.clip_sample: bool = config.get("clip_sample", True)
        self.clip_sample_range: float = config.get("clip_sample_range", 1.0)

        self.betas = _compute_betas(
            self.num_train_timesteps,
            config.get("beta_start", 0.0001),
            config.get("beta_end", 0.02),
            config.get("beta_schedule", "squaredcos_cap_v2"),
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.num_inference_steps: int | None = None

    def set_timesteps(self, num_inference_steps: int) -> NDArray[np.int64]:
        """Set the number of inference steps and return the timestep schedule.

        Args:
            num_inference_steps: Number of denoising steps.

        Returns:
            Array of timestep indices in reverse order.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        return (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)

    def step(
        self,
        model_output: NDArray[np.floating],
        timestep: int,
        sample: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Perform one DDPM denoising step.

        Args:
            model_output: Model prediction (noise, sample, or v-prediction).
            timestep: Current timestep index.
            sample: Current noisy sample.

        Returns:
            Denoised sample for the previous timestep.

        Raises:
            ValueError: If ``set_timesteps`` was not called first, or if
                ``prediction_type`` is unknown.
        """
        if self.num_inference_steps is None:
            msg = "Must call set_timesteps before step"
            raise ValueError(msg)

        t = timestep
        prev_t = t - self.num_train_timesteps // self.num_inference_steps

        alpha_prod_t = self.alphas_cumprod[t]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else np.float32(1.0)
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        current_alpha_t = alpha_prod_t / alpha_prod_t_prev
        current_beta_t = 1 - current_alpha_t

        if self.prediction_type == "epsilon":
            pred_original = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
        elif self.prediction_type == "sample":
            pred_original = model_output
        elif self.prediction_type == "v_prediction":
            pred_original = np.sqrt(alpha_prod_t) * sample - np.sqrt(beta_prod_t) * model_output
        else:
            msg = f"Unknown prediction_type: {self.prediction_type}"
            raise ValueError(msg)

        if self.clip_sample:
            pred_original = np.clip(pred_original, -self.clip_sample_range, self.clip_sample_range)

        coeff_original = (np.sqrt(alpha_prod_t_prev) * current_beta_t) / beta_prod_t
        coeff_current = np.sqrt(current_alpha_t) * beta_prod_t_prev / beta_prod_t
        pred_prev = coeff_original * pred_original + coeff_current * sample

        if t > 0:
            variance = (beta_prod_t_prev / beta_prod_t) * current_beta_t
            variance = np.clip(variance, 1e-20, None)
            noise = np.random.default_rng().standard_normal(model_output.shape).astype(np.float32)
            pred_prev += np.sqrt(variance) * noise

        return pred_prev


class DDIMScheduler:
    """Denoising Diffusion Implicit Model scheduler (pure numpy).

    Args:
        config: Scheduler configuration dict.
        eta: Stochasticity parameter (0 = deterministic DDIM).
    """

    def __init__(self, config: dict[str, Any], eta: float = 0.0) -> None:
        self.num_train_timesteps: int = config.get("num_train_timesteps", 1000)
        self.prediction_type: str = config.get("prediction_type", "epsilon")
        self.clip_sample: bool = config.get("clip_sample", True)
        self.clip_sample_range: float = config.get("clip_sample_range", 1.0)
        self.eta = eta

        self.betas = _compute_betas(
            self.num_train_timesteps,
            config.get("beta_start", 0.0001),
            config.get("beta_end", 0.02),
            config.get("beta_schedule", "squaredcos_cap_v2"),
        )
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas, axis=0)
        self.final_alpha_cumprod = np.float32(1.0)
        self.num_inference_steps: int | None = None

    def set_timesteps(self, num_inference_steps: int) -> NDArray[np.int64]:
        """Set the number of inference steps and return the timestep schedule.

        Args:
            num_inference_steps: Number of denoising steps.

        Returns:
            Array of timestep indices in reverse order.
        """
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_train_timesteps // num_inference_steps
        return (np.arange(0, num_inference_steps) * step_ratio).round()[::-1].astype(np.int64)

    def step(
        self,
        model_output: NDArray[np.floating],
        timestep: int,
        sample: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Perform one DDIM denoising step.

        Args:
            model_output: Model prediction (noise, sample, or v-prediction).
            timestep: Current timestep index.
            sample: Current noisy sample.

        Returns:
            Denoised sample for the previous timestep.

        Raises:
            ValueError: If ``set_timesteps`` was not called first, or if
                ``prediction_type`` is unknown.
        """
        if self.num_inference_steps is None:
            msg = "Must call set_timesteps before step"
            raise ValueError(msg)

        prev_t = timestep - self.num_train_timesteps // self.num_inference_steps
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_t] if prev_t >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t

        if self.prediction_type == "epsilon":
            pred_original = (sample - np.sqrt(beta_prod_t) * model_output) / np.sqrt(alpha_prod_t)
            pred_epsilon = model_output
        elif self.prediction_type == "sample":
            pred_original = model_output
            pred_epsilon = (sample - np.sqrt(alpha_prod_t) * pred_original) / np.sqrt(beta_prod_t)
        elif self.prediction_type == "v_prediction":
            pred_original = np.sqrt(alpha_prod_t) * sample - np.sqrt(beta_prod_t) * model_output
            pred_epsilon = np.sqrt(alpha_prod_t) * model_output + np.sqrt(beta_prod_t) * sample
        else:
            msg = f"Unknown prediction_type: {self.prediction_type}"
            raise ValueError(msg)

        if self.clip_sample:
            pred_original = np.clip(pred_original, -self.clip_sample_range, self.clip_sample_range)

        variance = ((1 - alpha_prod_t_prev) / (1 - alpha_prod_t)) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = self.eta * np.sqrt(variance)

        pred_sample_direction = np.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * pred_epsilon
        prev_sample = np.sqrt(alpha_prod_t_prev) * pred_original + pred_sample_direction

        if self.eta > 0:
            noise = np.random.default_rng().standard_normal(model_output.shape).astype(np.float32)
            prev_sample += std_dev_t * noise

        return prev_sample
