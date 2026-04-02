# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ONNX Runtime adapter for inference."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from physicalai.inference.adapters.base import RuntimeAdapter

if TYPE_CHECKING:
    import onnxruntime

_ONNX_TYPE_TO_NUMPY = {
    "tensor(float)": np.float32,
    "tensor(float16)": np.float16,
    "tensor(double)": np.float64,
    "tensor(int64)": np.int64,
    "tensor(int32)": np.int32,
    "tensor(int8)": np.int8,
    "tensor(uint8)": np.uint8,
    "tensor(bool)": np.bool_,
}


class ONNXAdapter(RuntimeAdapter):
    """ONNX Runtime inference adapter.

    Supports CPU and GPU acceleration.  Auto-casts input dtypes to
    match ONNX graph expectations and filters inputs to only the
    names the model declares.

    Examples:
        >>> adapter = ONNXAdapter(device="cpu", model_path=Path("model.onnx"))
        >>> outputs = adapter.predict({"input": input_array})
    """

    def __init__(
        self,
        device: str = "cpu",
        model_path: Path | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> None:
        """Initialize ONNX adapter.

        Args:
            device: Device for inference (``"cpu"``, ``"cuda"``, ``"tensorrt"``).
            model_path: Optional ONNX path to eagerly load.
            **kwargs: Additional ONNX Runtime adapter configuration.
        """
        super().__init__(device, **kwargs)
        self.session: onnxruntime.InferenceSession | None = None
        self._input_names: list[str] = []
        self._output_names: list[str] = []
        self._input_metadata: dict[str, dict[str, Any]] = {}
        if model_path is not None:
            self.load(model_path)

    def load(self, model_path: Path) -> None:
        """Load ONNX model from file.

        Args:
            model_path: Path to ``.onnx`` model file.

        Raises:
            ImportError: If ``onnxruntime`` is not installed.
            FileNotFoundError: If model file does not exist.
        """
        try:
            import onnxruntime as ort  # noqa: PLC0415
        except ImportError as e:
            msg = "ONNX Runtime is not installed. Install with: uv pip install onnxruntime"
            raise ImportError(msg) from e

        model_path = Path(model_path)
        if not model_path.exists():
            msg = f"Model file not found: {model_path}"
            raise FileNotFoundError(msg)

        providers = self._get_providers()
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(str(model_path), sess_options=sess_options, providers=providers)

        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]
        self._input_metadata = {i.name: {"shape": i.shape, "dtype": i.type} for i in self.session.get_inputs()}

    def predict(self, inputs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Run inference with ONNX Runtime.

        Args:
            inputs: Dictionary mapping input names to arrays.

        Returns:
            Dictionary mapping output names to arrays.

        Raises:
            RuntimeError: If model is not loaded.
            ValueError: If required inputs are missing.
        """
        if self.session is None:
            msg = "Model not loaded. Call load() first."
            raise RuntimeError(msg)

        ort_inputs = {k: v for k, v in inputs.items() if k in self._input_names}
        missing = set(self._input_names) - set(ort_inputs.keys())
        if missing:
            msg = f"Missing required inputs: {missing}"
            raise ValueError(msg)

        for name, arr in ort_inputs.items():
            expected_type = self._input_metadata[name]["dtype"]
            np_dtype = _ONNX_TYPE_TO_NUMPY.get(expected_type)
            if np_dtype is not None and arr.dtype != np_dtype:
                ort_inputs[name] = arr.astype(np_dtype)

        raw_outputs = self.session.run(self._output_names, ort_inputs)
        outputs = cast("list[np.ndarray]", raw_outputs)
        return dict(zip(self._output_names, outputs, strict=True))

    def _get_providers(self) -> list[str | tuple[str, dict[str, Any]]]:
        """Get ONNX Runtime providers based on selected device.

        Returns:
            Providers in priority order.
        """
        device_lower = self.device.lower()

        if device_lower.startswith("cuda") or device_lower == "gpu":
            device_id = 0
            if ":" in device_lower:
                try:
                    device_id = int(device_lower.split(":")[1])
                except (ValueError, IndexError):
                    device_id = 0
            return [("CUDAExecutionProvider", {"device_id": device_id}), "CPUExecutionProvider"]
        if device_lower == "tensorrt":
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]

        return ["CPUExecutionProvider"]

    def default_device(self) -> str:  # noqa: PLR6301
        """Get default ONNX Runtime device.

        Returns:
            ``"cpu"``.
        """
        return "cpu"

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
