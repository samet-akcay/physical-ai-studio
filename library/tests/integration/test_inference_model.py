# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Integration tests for InferenceModel with lerobot-exported policies.

These tests require lerobot to be installed and exercise the full
export → load → inference roundtrip for ACT, Diffusion, and PI0.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

onnxruntime = pytest.importorskip("onnxruntime")
torch = pytest.importorskip("torch")


def _skip_if_no_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


def _skip_if_pi0_unavailable() -> None:
    pytest.importorskip("transformers")


def _to_numpy(batch: dict) -> dict[str, np.ndarray]:
    return {k: v.detach().cpu().numpy() for k, v in batch.items()}


def _export_act(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray], int, int]:
    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.export import export_policy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.act.modeling_act import ACTPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    config = ACTConfig(
        device="cpu",
        chunk_size=10,
        n_action_steps=10,
        dim_model=64,
        n_heads=2,
        dim_feedforward=128,
        n_encoder_layers=2,
        n_decoder_layers=2,
        n_vae_encoder_layers=2,
        use_vae=False,
        latent_dim=16,
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        },
    )

    policy = ACTPolicy(config)
    policy.eval()

    batch = {
        "observation.state": torch.randn(1, 6),
        "observation.images.top": torch.randn(1, 3, 84, 84),
    }

    package_path = export_policy(
        policy,
        tmp_path / "act_package",
        backend="onnx",
        example_batch=batch,
        include_normalization=False,
    )

    return package_path, _to_numpy(batch), config.chunk_size, 6


def _export_diffusion(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray], int, int]:
    pytest.importorskip("diffusers")

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.export import export_policy
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    from lerobot.utils.constants import ACTION, OBS_STATE

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 84, 84)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(6,)),
    }

    config = DiffusionConfig(
        device="cpu",
        n_action_steps=8,
        horizon=8,
        n_obs_steps=2,
        num_inference_steps=5,
        down_dims=(64, 128),
        vision_backbone="resnet18",
        pretrained_backbone_weights=None,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MIN_MAX,
            "ACTION": NormalizationMode.MIN_MAX,
        },
        noise_scheduler_type="DDIM",
    )

    policy = DiffusionPolicy(config)
    policy.eval()

    batch = {
        "observation.state": torch.randn(1, 2, 6),
        "observation.images.top": torch.randn(1, 2, 3, 84, 84),
    }

    package_path = export_policy(
        policy,
        tmp_path / "diffusion_package",
        backend="onnx",
        example_batch=batch,
        include_normalization=False,
    )

    return package_path, _to_numpy(batch), config.horizon, 6


def _export_pi0(tmp_path: Path) -> tuple[Path, dict[str, np.ndarray], int, int]:
    _skip_if_pi0_unavailable()
    _skip_if_no_cuda()

    from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
    from lerobot.export import export_policy
    from lerobot.policies.pi0.configuration_pi0 import PI0Config
    from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    from lerobot.utils.constants import (
        ACTION,
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
        OBS_STATE,
    )

    input_features = {
        OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(14,)),
        "observation.images.top": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
    }
    output_features = {
        ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(14,)),
    }

    config = PI0Config(
        device="cuda",
        chunk_size=10,
        n_action_steps=10,
        max_state_dim=32,
        max_action_dim=32,
        num_inference_steps=3,
        tokenizer_max_length=48,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
        },
        freeze_vision_encoder=True,
        train_expert_only=True,
        dtype="float32",
    )

    policy = PI0Policy(config)
    policy.to("cuda")
    policy.eval()

    batch = {
        OBS_STATE: torch.randn(1, 14, device="cuda"),
        "observation.images.top": torch.rand(1, 3, 224, 224, device="cuda"),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 48, dtype=torch.long, device="cuda"),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 48, dtype=torch.bool, device="cuda"),
    }

    package_path = export_policy(
        policy,
        tmp_path / "pi0_package",
        backend="onnx",
        example_batch=batch,
        include_normalization=False,
    )

    obs_numpy = _to_numpy(batch)
    return package_path, obs_numpy, config.chunk_size, 32


class TestInferenceModelACT:
    """ACT policy (single-pass + action chunking) integration tests."""

    @pytest.mark.slow
    def test_load_and_run(self, tmp_path: Path) -> None:
        from physicalai.inference import InferenceModel

        package_path, obs, _chunk_size, action_dim = _export_act(tmp_path)

        model = InferenceModel(package_path, backend="onnx", device="cpu")
        assert model.manifest is not None
        assert model.manifest.policy.name is not None

        outputs = model(obs)
        assert "action" in outputs
        action = outputs["action"]
        assert action.shape[-1] == action_dim

    @pytest.mark.slow
    def test_select_action_returns_chunk_sequentially(self, tmp_path: Path) -> None:
        from physicalai.inference import InferenceModel

        package_path, obs, chunk_size, action_dim = _export_act(tmp_path)

        model = InferenceModel(package_path, backend="onnx", device="cpu")

        actions = []
        for _ in range(chunk_size):
            action = model.select_action(obs)
            assert isinstance(action, np.ndarray)
            assert action.shape[-1] == action_dim
            actions.append(action)

        assert len(actions) == chunk_size

    @pytest.mark.slow
    def test_reset(self, tmp_path: Path) -> None:
        from physicalai.inference import InferenceModel

        package_path, _obs, _, _ = _export_act(tmp_path)
        model = InferenceModel(package_path, backend="onnx", device="cpu")
        model.reset()

    @pytest.mark.slow
    def test_context_manager(self, tmp_path: Path) -> None:
        from physicalai.inference import InferenceModel

        package_path, obs, _, action_dim = _export_act(tmp_path)

        with InferenceModel(package_path, backend="onnx", device="cpu") as model:
            action = model.select_action(obs)
            assert action.shape[-1] == action_dim

    @pytest.mark.slow
    def test_openvino_backend(self, tmp_path: Path) -> None:
        pytest.importorskip("openvino")
        from physicalai.inference import InferenceModel

        package_path, obs, _chunk_size, action_dim = _export_act(tmp_path)

        model = InferenceModel(package_path, backend="openvino", device="CPU")
        outputs = model(obs)
        assert "action" in outputs
        action = outputs["action"]
        assert action.shape[-1] == action_dim

    @pytest.mark.slow
    def test_onnx_openvino_parity(self, tmp_path: Path) -> None:
        pytest.importorskip("openvino")
        from physicalai.inference import InferenceModel

        package_path, obs, _, _ = _export_act(tmp_path)

        onnx_model = InferenceModel(package_path, backend="onnx", device="cpu")
        ov_model = InferenceModel(package_path, backend="openvino", device="CPU")

        onnx_out = onnx_model(obs)["action"]
        ov_out = ov_model(obs)["action"]

        np.testing.assert_allclose(onnx_out, ov_out, rtol=1e-4, atol=1e-5)


class TestInferenceModelDiffusion:
    """Diffusion policy (iterative runner) integration tests."""

    @pytest.mark.slow
    def test_load_and_run(self, tmp_path: Path) -> None:
        from physicalai.inference import InferenceModel

        package_path, obs, chunk_size, action_dim = _export_diffusion(tmp_path)

        model = InferenceModel(package_path, backend="onnx", device="cpu")
        assert model.manifest is not None

        outputs = model(obs)
        assert "action" in outputs
        action = outputs["action"]
        assert action.shape[-2] == chunk_size
        assert action.shape[-1] == action_dim


class TestInferenceModelPI0:
    """PI0 VLA policy (two-phase runner) integration tests.

    NOTE: PI0 export requires a patched transformers build with SigLIP
    replace hooks. If the installed version lacks these, the KV cache
    will be empty (not a physicalai issue).
    """

    @pytest.mark.slow
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="PI0 export requires CUDA",
    )
    def test_load_and_run(self, tmp_path: Path) -> None:
        from physicalai.inference import InferenceModel

        try:
            package_path, obs, chunk_size, action_dim = _export_pi0(tmp_path)
        except (AttributeError, IndexError) as exc:
            pytest.skip(f"PI0 export requires patched transformers (SigLIP hooks): {exc}")

        model = InferenceModel(package_path, backend="onnx", device="cpu")
        assert model.manifest is not None

        outputs = model(obs)
        assert "action" in outputs
        action = outputs["action"]
        assert action.shape[-2] == chunk_size
        assert action.shape[-1] == action_dim


class TestInferenceModelManifestCompat:
    """Manifest loading and compatibility tests."""

    @pytest.mark.slow
    def test_manifest_fields_loaded(self, tmp_path: Path) -> None:
        from physicalai.inference import InferenceModel

        package_path, _, _, _ = _export_act(tmp_path)
        model = InferenceModel(package_path, backend="onnx", device="cpu")

        m = model.manifest
        assert m.format == "policy_package"
        assert m.version.startswith("1.")
        assert m.model.runner.type == "action_chunking"
        assert "model" in m.model.artifacts

    @pytest.mark.slow
    def test_repr(self, tmp_path: Path) -> None:
        from physicalai.inference import InferenceModel

        package_path, _, _, _ = _export_act(tmp_path)
        model = InferenceModel(package_path, backend="onnx", device="cpu")

        r = repr(model)
        assert "InferenceModel" in r
        assert "onnx" in r
