# Policy Runtime Design

> **Merged.** This document has been merged into [`robot_runtime_architecture.md`](./robot_runtime_architecture.md).
>
> The policy-deployment runtime is no longer a separate design. It is one `Controller` (`PolicyController`) running inside `RobotRuntime`. The convenience entry point `PolicyRuntime` is now a one-line factory over `RobotRuntime + PolicyController` and is documented in §11 of the architecture doc.

## Where things moved

| Topic                                                     | New location                                                                            |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `PolicyRuntime` API and example                           | [`robot_runtime_architecture.md` §11](./robot_runtime_architecture.md#11-policyruntime-convenience-factory) |
| Component ownership table                                 | [`robot_runtime_architecture.md` §5](./robot_runtime_architecture.md#5-policycontroller-and-inference)     |
| `select_action()` vs `predict_action_chunk()`             | [`robot_runtime_architecture.md` §5](./robot_runtime_architecture.md#5-policycontroller-and-inference)     |
| Chunking, queueing, `ActionChunkCursor`, `ActionQueue`    | [`robot_runtime_architecture.md` §5](./robot_runtime_architecture.md#5-policycontroller-and-inference)     |
| Execution modes (sync / thread / process / remote)        | [`robot_runtime_architecture.md` §5](./robot_runtime_architecture.md#inferenceexecution-modes)             |
| Loop body and lifecycle                                   | [`robot_runtime_architecture.md` §8](./robot_runtime_architecture.md#8-interfaces)                          |
| Workflows (HIL, recording, DAgger, highlight, teleop)     | [`robot_runtime_architecture.md` §6](./robot_runtime_architecture.md#6-workflows)                           |
| Config examples and CLI                                   | [`robot_runtime_architecture.md` §12](./robot_runtime_architecture.md#12-config-and-cli)                    |
| Implementation phases                                     | [`robot_runtime_architecture.md` §16](./robot_runtime_architecture.md#16-implementation-phases)             |
| Benchmarking vs runtime                                   | [`robot_runtime_architecture.md` §13](./robot_runtime_architecture.md#13-benchmarking-vs-runtime)           |
| Multi-system autonomy (VLA + locomotion + planner + …)   | [`composite_robot_architecture.md`](./composite_robot_architecture.md)                  |
| Remote inference / `PolicyServer` / `RemoteExecution`     | [`policy_server_design.md`](./policy_server_design.md)                                  |
| Action / observation evolution (`np.ndarray` → mapping)   | [`../robot-interface.md`](../robot-interface.md)                                        |

Update any links pointing at this file to the corresponding section above.
