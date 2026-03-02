## Task 3: Decouple model.py from torch runtime

### Key Findings
- `physicalai.data.__init__` imports `DataModule` from `datamodules.py` which does `import torch` at top level
- This means `from physicalai.data.constants import ACTION, IMAGES, STATE` transitively triggers torch import via `physicalai.data.__init__.py`
- The verification command `assert 'torch' not in sys.modules` cannot pass until `physicalai.data.__init__` is also fixed (separate task)
- model.py itself is fully decoupled: no top-level torch, all torch ops replaced with numpy equivalents

### Changes Made
- `import numpy as np` moved from TYPE_CHECKING to top-level (it was incorrectly in TYPE_CHECKING despite being a runtime dep)
- `deque[torch.Tensor]` → `deque[np.ndarray]`
- `select_action()` return type: `torch.Tensor` → `np.ndarray`, accepts `Observation | dict[str, np.ndarray]`
- All torch tensor ops replaced: `torch.from_numpy()` removed, `.transpose()` → `np.transpose()`, `.dim()` → `.ndim`, `.squeeze()` → `np.squeeze()`
- Added `_backward_compat_wrap()` for backward compat: Observation callers get torch.Tensor, dict callers get np.ndarray
- `_prepare_inputs()` accepts dict[str, np.ndarray] with early return, removed local numpy import

### Test Results
- All 76 tests in `tests/unit/inference/` pass
