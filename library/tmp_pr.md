## Summary

- add torch-export and load support for LeRobot wrappers so exported checkpoints retain LeRobot config/state and can be reloaded through `InferenceModel(backend="torch")`
- extend LeRobot wrapper equivalence integration coverage to compare exported torch inference against both the wrapped LeRobot policy and a native LeRobot replay model on the same input
- calibrate deterministic inference checks for stochastic/VLA policies and verify GPU parity locally for `act`, `smolvla`, and `pi05`

## Testing

- `uv run pytest 'tests/integration/test_lerobot_wrapper_equivalence.py::TestTorchInferenceEquivalence::test_torch_inference_matches_native_lerobot[act]' -q`
- `uv run pytest 'tests/integration/test_lerobot_wrapper_equivalence.py::TestTorchInferenceEquivalence::test_torch_inference_matches_native_lerobot[smolvla]' -q`
- `uv run pytest 'tests/integration/test_lerobot_wrapper_equivalence.py::TestTorchInferenceEquivalence::test_torch_inference_matches_native_lerobot[pi05]' -q`
- `uv run pytest tests/unit/inference/test_adapters.py::TestTorchAdapter tests/unit/policies/test_lerobot.py::TestLeRobotPolicyCheckpoint::test_load_from_checkpoint_universal_wrapper -q`
