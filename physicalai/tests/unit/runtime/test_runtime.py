# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from unittest.mock import MagicMock

import numpy as np
import pytest

from physicalai.runtime import (
    ActionQueue,
    AsyncInferenceExecution,
    HoldStateFallback,
    PolicyController,
    PolicyRuntime,
    RobotRuntime,
    RuntimeCallback,
    SyncInferenceExecution,
)
from physicalai.runtime.safety import SafetyViolationError


@dataclass
class FakeObservation:
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros(6))
    timestamp: float = 0.0
    sensor_data: dict | None = None
    images: dict | None = None


class FakeRobot:
    def __init__(self, num_joints: int = 6) -> None:
        self._connected = False
        self._num_joints = num_joints
        self.actions_received: list[np.ndarray] = []

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_observation(self) -> FakeObservation:
        return FakeObservation(
            joint_positions=np.zeros(self._num_joints),
            timestamp=time.monotonic(),
        )

    def send_action(self, action: np.ndarray) -> None:
        self.actions_received.append(action.copy())

    def is_connected(self) -> bool:
        return self._connected

    @property
    def joint_names(self) -> list[str]:
        return [f"joint_{i}" for i in range(self._num_joints)]


class ConstantController:
    def __init__(self, action: np.ndarray) -> None:
        self._action = action

    def start(self) -> None:
        pass

    def update(self, observation: dict) -> np.ndarray:
        return self._action.copy()

    def stop(self) -> None:
        pass

    def reset(self) -> None:
        pass


class TestActionQueue:
    def test_push_and_pop(self) -> None:
        queue = ActionQueue()
        chunk = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        queue.push_chunk(chunk)

        assert len(queue) == 3
        np.testing.assert_array_equal(queue.pop_or_none(), [1.0, 2.0])
        np.testing.assert_array_equal(queue.pop_or_none(), [3.0, 4.0])
        np.testing.assert_array_equal(queue.pop_or_none(), [5.0, 6.0])
        assert queue.pop_or_none() is None

    def test_clear(self) -> None:
        queue = ActionQueue()
        queue.push_chunk(np.ones((5, 2)))
        queue.clear()
        assert queue.empty


class TestRobotRuntime:
    def test_runs_for_duration(self) -> None:
        robot = FakeRobot()
        action = np.ones(6)
        controller = ConstantController(action)

        runtime = RobotRuntime(robot=robot, controller=controller, fps=100)
        runtime.run(duration_s=0.05)

        assert len(robot.actions_received) >= 3
        np.testing.assert_array_equal(robot.actions_received[0], action)

    def test_stop_from_thread(self) -> None:
        robot = FakeRobot()
        controller = ConstantController(np.zeros(6))
        runtime = RobotRuntime(robot=robot, controller=controller, fps=100)

        def stop_after_delay() -> None:
            time.sleep(0.03)
            runtime.stop()

        t = threading.Thread(target=stop_after_delay)
        t.start()
        runtime.run()
        t.join()
        assert not runtime.running

    def test_callback_on_observation(self) -> None:
        robot = FakeRobot()
        controller = ConstantController(np.zeros(6))
        cb = RuntimeCallback()
        cb.on_observation = MagicMock()

        runtime = RobotRuntime(robot=robot, controller=controller, fps=100, callbacks=[cb])
        runtime.run(duration_s=0.02)

        assert cb.on_observation.call_count >= 1

    def test_safety_violation_stops_loop(self) -> None:
        robot = FakeRobot()
        controller = ConstantController(np.zeros(6))

        class ViolatingSafety:
            def filter(self, action, observation):
                raise SafetyViolationError("unsafe")

        runtime = RobotRuntime(robot=robot, controller=controller, fps=100, safety=ViolatingSafety())
        runtime.run(duration_s=1.0)

        assert len(robot.actions_received) == 0

    def test_controller_error_holds_last_action(self) -> None:
        robot = FakeRobot()
        call_count = 0

        class FailingController:
            def start(self):
                pass

            def update(self, observation):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return np.ones(6)
                raise RuntimeError("fail")

            def stop(self):
                pass

            def reset(self):
                pass

        runtime = RobotRuntime(robot=robot, controller=FailingController(), fps=100)
        runtime.run(duration_s=0.05)

        assert len(robot.actions_received) >= 2
        np.testing.assert_array_equal(robot.actions_received[0], np.ones(6))
        np.testing.assert_array_equal(robot.actions_received[1], np.ones(6))

    def test_swap_controller(self) -> None:
        robot = FakeRobot()
        ctrl1 = ConstantController(np.ones(6))
        ctrl2 = ConstantController(np.full(6, 2.0))

        runtime = RobotRuntime(robot=robot, controller=ctrl1, fps=100)

        def swap_mid_run() -> None:
            time.sleep(0.02)
            runtime.swap_controller(ctrl2)
            time.sleep(0.02)
            runtime.stop()

        t = threading.Thread(target=swap_mid_run)
        t.start()
        runtime.run()
        t.join()

        has_twos = any(np.allclose(a, 2.0) for a in robot.actions_received)
        assert has_twos


class TestPolicyController:
    def test_sync_single_action(self) -> None:
        model = MagicMock()
        model.select_action = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        model.reset = MagicMock()

        execution = SyncInferenceExecution(mode="single_action")
        pc = PolicyController(model=model, execution=execution)
        pc.start()

        obs = {"state": np.zeros(3)}
        action = pc.update(obs)
        np.testing.assert_array_equal(action, [1.0, 2.0, 3.0])

    def test_sync_chunk_mode(self) -> None:
        chunk = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = MagicMock()
        model.return_value = {"action": chunk}
        model.reset = MagicMock()

        execution = SyncInferenceExecution(mode="chunk")
        pc = PolicyController(model=model, execution=execution)
        pc.start()

        obs = {"state": np.zeros(2)}
        a1 = pc.update(obs)
        np.testing.assert_array_equal(a1, [1.0, 2.0])
        a2 = pc.update(obs)
        np.testing.assert_array_equal(a2, [3.0, 4.0])

    def test_reset_clears_queue(self) -> None:
        chunk = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = MagicMock()
        model.return_value = {"action": chunk}
        model.reset = MagicMock()

        execution = SyncInferenceExecution(mode="chunk")
        pc = PolicyController(model=model, execution=execution)
        pc.start()

        obs = {"state": np.zeros(2)}
        pc.update(obs)
        pc.reset()

        model.return_value = {"action": np.array([[9.0, 9.0]])}
        a = pc.update(obs)
        np.testing.assert_array_equal(a, [9.0, 9.0])


class TestPolicyControllerFallback:
    @staticmethod
    def _noop_execution():
        class NoopExecution:
            def start(self, action_queue, model) -> None:
                pass

            def maybe_request(self, observation) -> None:
                pass

            def warmup(self, sample_observation, n=2) -> None:
                pass

            def stop(self) -> None:
                pass

        return NoopExecution()

    def test_no_action_no_fallback_raises(self) -> None:
        model = MagicMock()
        model.reset = MagicMock()
        pc = PolicyController(model=model, execution=self._noop_execution())
        pc.start()

        with pytest.raises(RuntimeError, match="no action"):
            pc.update({"state": np.zeros(6)})

    def test_no_action_returns_last_action(self) -> None:
        chunk = np.array([[1.0, 2.0]])
        model = MagicMock()
        model.return_value = {"action": chunk}
        model.reset = MagicMock()

        execution = SyncInferenceExecution(mode="chunk")
        pc = PolicyController(model=model, execution=execution)
        pc.start()

        obs = {"state": np.zeros(2)}
        first = pc.update(obs)
        np.testing.assert_array_equal(first, [1.0, 2.0])

        pc._execution = self._noop_execution()
        held = pc.update(obs)
        np.testing.assert_array_equal(held, [1.0, 2.0])

    def test_fallback_used_when_queue_and_last_action_empty(self) -> None:
        model = MagicMock()
        model.reset = MagicMock()
        fallback = HoldStateFallback()
        pc = PolicyController(
            model=model,
            execution=self._noop_execution(),
            fallback=fallback,
        )
        pc.start()

        state = np.array([0.1, 0.2, 0.3])
        action = pc.update({"state": state})
        np.testing.assert_array_equal(action, state)
        assert pc.fallback_count == 1


class TestPolicyRuntime:
    def test_factory_creates_runtime(self) -> None:
        robot = FakeRobot()
        model = MagicMock()
        model.select_action = MagicMock(return_value=np.zeros(6))
        model.reset = MagicMock()

        execution = SyncInferenceExecution(mode="single_action")
        runtime = PolicyRuntime(robot=robot, model=model, execution=execution, fps=100)
        assert isinstance(runtime, RobotRuntime)

        runtime.run(duration_s=0.02)
        assert len(robot.actions_received) >= 1


class _MockModel:
    def __init__(self, chunk: np.ndarray, latency_s: float = 0.0, raises: BaseException | None = None) -> None:
        self._chunk = chunk
        self._latency_s = latency_s
        self._raises = raises
        self.call_count = 0
        self.observations_seen: list[dict] = []

    def __call__(self, observation: dict) -> dict:
        self.call_count += 1
        self.observations_seen.append(observation)
        if self._latency_s > 0:
            time.sleep(self._latency_s)
        if self._raises is not None:
            raise self._raises
        return {"action": self._chunk}

    def reset(self) -> None:
        pass


def _wait_for(predicate, timeout_s: float = 2.0, poll_s: float = 0.005) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(poll_s)
    return predicate()


class TestAsyncInferenceExecution:
    def test_bootstrap_empty_queue_with_fallback(self) -> None:
        chunk = np.array([[1.0, 2.0]])
        model = _MockModel(chunk, latency_s=0.05)
        execution = AsyncInferenceExecution(refill_threshold=0)
        pc = PolicyController(model=model, execution=execution, fallback=HoldStateFallback())
        pc.start()

        try:
            state = np.array([0.5, 0.5])
            action = pc.update({"state": state})
            np.testing.assert_array_equal(action, state)
            assert pc.fallback_count == 1
        finally:
            pc.stop()

    def test_steady_state_refill(self) -> None:
        chunk = np.tile(np.array([[1.0, 2.0]]), (4, 1))
        model = _MockModel(chunk, latency_s=0.02)
        execution = AsyncInferenceExecution(refill_threshold=2)
        queue = ActionQueue()
        pc = PolicyController(
            model=model, execution=execution, action_queue=queue,
            fallback=HoldStateFallback(),
        )
        pc.start()

        try:
            for _ in range(20):
                pc.update({"state": np.zeros(2)})
                time.sleep(0.01)
            assert model.call_count >= 2
            assert execution.inference_count >= 2
        finally:
            pc.stop()

    def test_latest_wins_observation_mailbox(self) -> None:
        chunk = np.array([[0.0, 0.0]])
        model = _MockModel(chunk, latency_s=0.05)
        execution = AsyncInferenceExecution(refill_threshold=0)
        queue = ActionQueue()

        execution.start(queue, model)
        try:
            # Submit 5 observations rapidly while the worker is busy on tag=0.
            # Drain queue after first inference so the worker is eligible to
            # refill; latest-wins mailbox should then pick tag=4, dropping 1-3.
            for i in range(5):
                execution.maybe_request({"state": np.zeros(2), "tag": i})
                time.sleep(0.001)
            assert _wait_for(lambda: execution.inference_count >= 1, timeout_s=2.0)
            while not queue.empty:
                queue.pop_or_none()
            execution.maybe_request({"state": np.zeros(2), "tag": 99})
            assert _wait_for(lambda: execution.inference_count >= 2, timeout_s=2.0)
            tags_seen = [obs["tag"] for obs in model.observations_seen]
            assert tags_seen[0] == 0, f"first processed tag should be 0, got {tags_seen[0]}"
            assert tags_seen[1] == 99, (
                f"latest maybe_request must overwrite mailbox (tags 1-4 dropped), "
                f"got second tag={tags_seen[1]}"
            )
        finally:
            execution.stop()

    def test_transient_failure_retries_then_recovers(self) -> None:
        chunk = np.array([[1.0, 1.0]])
        attempts = {"n": 0}
        def fail_then_recover(observation):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise TimeoutError("transient")
            return {"action": chunk}

        class _FlakyModel:
            call_count = 0
            def __call__(self, observation):
                _FlakyModel.call_count += 1
                return fail_then_recover(observation)
            def reset(self): pass

        execution = AsyncInferenceExecution(
            refill_threshold=0, backoff_schedule_s=(0.01, 0.01),
        )
        pc = PolicyController(model=_FlakyModel(), execution=execution, fallback=HoldStateFallback())
        pc.start()

        try:
            pc.update({"state": np.zeros(2)})
            assert _wait_for(lambda: execution.inference_count >= 1, timeout_s=2.0)
            assert execution.transient_failure_count == 1
        finally:
            pc.stop()

    def test_fatal_exception_surfaces_to_runtime_thread(self) -> None:
        model = _MockModel(np.array([[0.0]]), raises=ValueError("model corruption"))
        execution = AsyncInferenceExecution(refill_threshold=0)
        pc = PolicyController(model=model, execution=execution, fallback=HoldStateFallback())
        pc.start()

        try:
            pc.update({"state": np.zeros(1)})
            assert _wait_for(lambda: execution._fatal_exception is not None, timeout_s=2.0)
            with pytest.raises(ValueError, match="model corruption"):
                pc.update({"state": np.zeros(1)})
        finally:
            pc.stop()

    def test_consecutive_transient_failures_escalate_to_fatal(self) -> None:
        model = _MockModel(np.array([[0.0]]), raises=TimeoutError("always transient"))
        execution = AsyncInferenceExecution(
            refill_threshold=0, backoff_schedule_s=(0.001,),
            max_consecutive_failures=3,
        )
        pc = PolicyController(model=model, execution=execution, fallback=HoldStateFallback())
        pc.start()

        try:
            pc.update({"state": np.zeros(1)})
            assert _wait_for(
                lambda: execution.transient_failure_count >= 3 or execution._stop.is_set(),
                timeout_s=2.0,
            )
            assert execution.transient_failure_count >= 3
            assert _wait_for(
                lambda: execution._thread is None or not execution._thread.is_alive(),
                timeout_s=2.0,
            )
        finally:
            pc.stop()

    def test_stop_during_inference_completes_within_timeout(self) -> None:
        chunk = np.array([[1.0]])
        model = _MockModel(chunk, latency_s=0.3)
        execution = AsyncInferenceExecution(refill_threshold=0, shutdown_timeout_s=2.0)
        pc = PolicyController(model=model, execution=execution, fallback=HoldStateFallback())
        pc.start()

        pc.update({"state": np.zeros(1)})
        time.sleep(0.05)

        t0 = time.monotonic()
        pc.stop()
        elapsed = time.monotonic() - t0
        assert elapsed < 1.5, f"stop took {elapsed:.2f}s (worker may have orphaned)"

    def test_warmup_prefills_queue(self) -> None:
        chunk = np.array([[1.0, 2.0], [3.0, 4.0]])
        model = _MockModel(chunk, latency_s=0.0)
        queue = ActionQueue()
        execution = AsyncInferenceExecution(refill_threshold=0)
        execution.start(queue, model)

        try:
            execution.warmup({"state": np.zeros(2)}, n=2)
            assert len(queue) == 4
        finally:
            execution.stop()
