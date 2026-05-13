# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Optional MolmoAct2 policy integration.

MolmoAct2 currently lives in Ai2's LeRobot fork.  This module intentionally
does not vendor or reimplement the model; it provides a small PhysicalAI entry
point that delegates to the fork's LeRobot ``molmoact2`` policy when that fork
is installed or on ``PYTHONPATH``.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from physicalai.policies.lerobot.policy import NamedLeRobotPolicy

if TYPE_CHECKING:
    import torch

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import PolicyFeature


def _get_molmoact2_config_class() -> type[Any]:
    try:
        module = importlib.import_module("lerobot.policies.molmoact2.configuration_molmoact2")
    except ImportError as e:
        msg = (
            "MolmoAct2 requires Ai2's LeRobot fork from https://github.com/allenai/molmoact2. "
            "Install that fork or add its `lerobot/src` directory to PYTHONPATH before constructing "
            "physicalai.policies.molmoact2.MolmoAct2."
        )
        raise ImportError(msg) from e
    return module.MolmoAct2Config


class MolmoAct2(NamedLeRobotPolicy):
    """Thin PhysicalAI wrapper around Ai2's LeRobot ``molmoact2`` policy.

    The underlying MolmoAct2 policy is inference-only and performs image, state,
    language, normalization, and action scaling internally.  Use
    :meth:`from_checkpoint` or pass ``checkpoint_path=...`` directly.
    """

    POLICY_NAME = "molmoact2"

    def __init__(
        self,
        config: PreTrainedConfig | None = None,
        *,
        checkpoint_path: str | None = None,
        num_steps: int | None = None,
        action_mode: str = "continuous",
        discrete_action_tokenizer: str | None = None,
        discrete_generation_max_steps: int = 128,
        enable_depth_reasoning: bool = False,
        enable_adaptive_depth: bool = True,
        enable_cuda_graph: bool = True,
        normalize_language: bool = True,
        norm_tag: str = "",
        trust_remote_code: bool = True,
        input_features: dict[str, PolicyFeature] | None = None,
        output_features: dict[str, PolicyFeature] | None = None,
        dataset_stats: dict[str, dict[str, torch.Tensor]] | None = None,
        policy_config: dict[str, Any] | None = None,
        **overrides: Any,  # noqa: ANN401
    ) -> None:
        if config is None and checkpoint_path is not None:
            config_cls = _get_molmoact2_config_class()
            config = config_cls(
                checkpoint_path=checkpoint_path,
                num_steps=num_steps,
                action_mode=action_mode,
                discrete_action_tokenizer=discrete_action_tokenizer,
                discrete_generation_max_steps=discrete_generation_max_steps,
                enable_depth_reasoning=enable_depth_reasoning,
                enable_adaptive_depth=enable_adaptive_depth,
                enable_cuda_graph=enable_cuda_graph,
                normalize_language=normalize_language,
                norm_tag=norm_tag,
                trust_remote_code=trust_remote_code,
                **overrides,
            )
            overrides = {}

        super().__init__(
            config=config,
            input_features=input_features,
            output_features=output_features,
            dataset_stats=dataset_stats,
            policy_config=policy_config,
            **overrides,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        *,
        norm_tag: str,
        num_steps: int | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> MolmoAct2:
        """Create MolmoAct2 from a MolmoAct2 checkpoint path or HF repo id.

        Args:
            checkpoint_path: Local model path or Hugging Face model repo id, for
                example ``"allenai/MolmoAct2-LIBERO"``.
            norm_tag: MolmoAct2 action-normalization tag required by the model.
            num_steps: Optional flow-matching inference steps.
            **kwargs: Additional MolmoAct2 config fields forwarded to the fork.

        Returns:
            Initialized MolmoAct2 policy wrapper.
        """
        return cls(checkpoint_path=checkpoint_path, norm_tag=norm_tag, num_steps=num_steps, **kwargs)

    def __repr__(self) -> str:
        config = getattr(self, "_config", None)
        checkpoint_path = getattr(config, "checkpoint_path", "") if config is not None else ""
        norm_tag = getattr(config, "norm_tag", "") if config is not None else ""
        return f"{self.__class__.__name__}(checkpoint_path={checkpoint_path!r}, norm_tag={norm_tag!r})"
