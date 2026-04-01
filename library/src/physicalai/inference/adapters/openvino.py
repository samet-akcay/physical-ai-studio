# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""OpenVINO adapter for inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np  # noqa: TC002

from physicalai.inference.adapters.base import RuntimeAdapter

if TYPE_CHECKING:
    import openvino

_DEVICE_MAP = {
    "cpu": "CPU",
    "gpu": "GPU",
    "npu": "NPU",
    "auto": "AUTO",
}


class OpenVINOAdapter(RuntimeAdapter):
    """OpenVINO inference adapter.

    Handles compiled-model input/output name remapping (compilation
    may rename tensor nodes) and uses ``InferRequest`` for efficient
    inference.

    Examples:
        >>> adapter = OpenVINOAdapter(device="CPU", model_path=Path("model.xml"))
        >>> outputs = adapter.predict({"input": input_array})
    """

    def __init__(
        self,
        device: str = "CPU",
        model_path: Path | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize OpenVINO adapter.

        Args:
            device: OpenVINO target device (``"CPU"``, ``"GPU"``, ``"NPU"``, ``"AUTO"``).
            model_path: Optional OpenVINO ``.xml`` path to eagerly load.
            **kwargs: Additional OpenVINO compile options.
        """
        super().__init__(device, **kwargs)
        self._compiled_model: openvino.CompiledModel | None = None
        self._infer_request: openvino.InferRequest | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._input_name_mapping: dict[str, str] = {}
        self._output_name_mapping: dict[str, str] = {}
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        """Load OpenVINO model from XML file.

        Args:
            model_path: Path to ``.xml`` model file (with sidecar ``.bin``).

        Raises:
            ImportError: If OpenVINO is not installed.
            FileNotFoundError: If model file does not exist.
        """
        try:
            import openvino as ov  # noqa: PLC0415
        except ImportError as e:
            msg = "OpenVINO is not installed. Install with: uv pip install openvino"
            raise ImportError(msg) from e

        model_path = Path(model_path)
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        core = ov.Core()
        model = core.read_model(model=str(model_path))

        original_input_names = [inp.get_any_name() for inp in model.inputs]
        original_output_names = [out.get_any_name() for out in model.outputs]

        ov_device = self._normalize_device()
        self._compiled_model = core.compile_model(model=model, device_name=ov_device, config=self.config)
        self._infer_request = self._compiled_model.create_infer_request()

        compiled_input_names = [inp.get_any_name() for inp in self._compiled_model.inputs]
        compiled_output_names = [out.get_any_name() for out in self._compiled_model.outputs]

        self._input_names = original_input_names
        self._output_names = original_output_names

        self._input_name_mapping = {
            orig: comp for orig, comp in zip(original_input_names, compiled_input_names, strict=True) if orig != comp
        }
        self._output_name_mapping = {
            orig: comp for orig, comp in zip(original_output_names, compiled_output_names, strict=True) if orig != comp
        }

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run inference with OpenVINO.

        Args:
            inputs: Dictionary mapping input names to arrays.

        Returns:
            Dictionary mapping output names to arrays.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If required inputs are missing.
        """
        if self._infer_request is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        ov_inputs = {k: v for k, v in inputs.items() if k in self._input_names}
        missing = set(self._input_names) - set(ov_inputs.keys())
        if missing:
            msg = f"Missing required inputs: {missing}"
            raise ValueError(msg)

        mapped_inputs: dict[str, Any] = {}
        for name, value in ov_inputs.items():
            compiled_name = self._input_name_mapping.get(name, name)
            mapped_inputs[compiled_name] = value

        self._infer_request.infer(mapped_inputs)

        outputs: dict[str, np.ndarray] = {}
        for orig_name in self._output_names:
            compiled_name = self._output_name_mapping.get(orig_name, orig_name)
            tensor = self._infer_request.get_tensor(compiled_name)
            outputs[orig_name] = tensor.data.copy()

        return outputs

    def default_device(self) -> str:  # noqa: PLR6301
        """Get default OpenVINO device.

        Returns:
            ``"CPU"``.
        """
        return "CPU"

    @property
    def input_names(self) -> list[str]:
        """Get input tensor names.

        Returns:
            List of input names.
        """
        return self._input_names

    @property
    def output_names(self) -> list[str]:
        """Get output tensor names.

        Returns:
            List of output names.
        """
        return self._output_names

    def _normalize_device(self) -> str:
        device = self.device.lower()
        if device.startswith(("cuda", "xpu")):
            return "GPU"
        return _DEVICE_MAP.get(device, device.upper())
