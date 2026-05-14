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
