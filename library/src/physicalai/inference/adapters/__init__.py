# Copyright (C) 2025-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Inference adapters for different backend runtimes."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from physicalai.export.backends import ExportBackend

from physicalai.inference.adapters.base import RuntimeAdapter
from physicalai.inference.adapters.onnx import ONNXAdapter
from physicalai.inference.adapters.openvino import OpenVINOAdapter

__all__ = [
    "ExecuTorchAdapter",  # noqa: F822
    "ONNXAdapter",
    "OpenVINOAdapter",
    "RuntimeAdapter",
    "TorchAdapter",  # noqa: F822
    "get_adapter",
]


def __getattr__(name: str) -> Any:  # noqa: ANN401
    """Lazy-load torch-dependent adapters on access.

    Args:
        name: Attribute name to load.

    Returns:
        The requested adapter class.

    Raises:
        AttributeError: If attribute is not found.
    """
    if name == "TorchAdapter":
        from physicalai.inference.adapters.torch import TorchAdapter  # noqa: PLC0415

        return TorchAdapter
    if name == "ExecuTorchAdapter":
        from physicalai.inference.adapters.executorch import ExecuTorchAdapter  # noqa: PLC0415

        return ExecuTorchAdapter
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


def get_adapter(  # noqa: RUF067
    backend: ExportBackend | str,
    model_path: str | Path | None = None,
    device: str = "auto",
    **kwargs: Any,  # noqa: ANN401
) -> RuntimeAdapter:
    """Get the appropriate adapter for a given backend.

    When *model_path* is provided the adapter is constructed with the
    path and auto-loads the model immediately.  When *model_path* is
    ``None`` the caller must call ``adapter.load(path)`` manually
    (backward-compatible code path).

    Args:
        backend: The export backend (ExportBackend enum or string,
            e.g. ``"onnx"``, ``"openvino"``).
        model_path: Optional path to the model file.  When given, the
            adapter is constructed with ``model_path=`` and loads
            the model eagerly.
        device: Device for inference (``"auto"``, ``"cpu"``,
            ``"cuda"``, ``"CPU"``, ``"GPU"``, etc.).  Passed to the
            adapter constructor as ``device=``.
        **kwargs: Additional adapter configuration forwarded to the
            adapter constructor.

    Returns:
        Instantiated (and optionally loaded) adapter for the backend.

    Raises:
        ValueError: If backend is not supported.

    Examples:
        >>> from physicalai.export.backends import ExportBackend
        >>> adapter = get_adapter(ExportBackend.OPENVINO, device="CPU")
        >>> adapter.load(Path("model.xml"))
        >>> adapter = get_adapter("onnx", model_path=Path("model.onnx"))
    """
    backend_str = backend.value if hasattr(backend, "value") else str(backend)

    adapter_map: dict[str, type[RuntimeAdapter]] = {
        "openvino": OpenVINOAdapter,
        "onnx": ONNXAdapter,
    }

    if backend_str == "torch":
        from physicalai.inference.adapters.torch import TorchAdapter  # noqa: PLC0415

        adapter_map["torch"] = TorchAdapter

    if backend_str == "executorch":
        from physicalai.inference.adapters.executorch import ExecuTorchAdapter  # noqa: PLC0415

        adapter_map["executorch"] = ExecuTorchAdapter

    if backend_str not in adapter_map:
        msg = f"No adapter available for backend: {backend_str}"
        raise ValueError(msg)

    ctor_kwargs: dict[str, Any] = {"device": device, **kwargs}
    if model_path is not None:
        ctor_kwargs["model_path"] = Path(model_path)

    return adapter_map[backend_str](**ctor_kwargs)
