# Reference bridge (HTTP + stub ROS)

Small **samples** for connecting an on-robot process to Haroma’s **executor contract**:

- **`haroma_client.py`** — POST `/robot/bridge/feedback` and GET `/status` (stdlib `urllib` only).
- **`stub_executor.py`** — Turn a `build_executor_command_batch` payload into `results[]` (replace with your controller).
- **`ros2_stub.py`** — Optional `rclpy` import probe + `StubExecutorNode` helper (no mandatory ROS dependency).
- **`sample_http_bridge.py`** — End-to-end demo against a running Haroma server.

**Step-by-step:** [Robot integration (step-by-step)](../docs/robot-integration.md). **Concepts & safety:** [Robot cognitive / control split](../docs/robot-cognitive-control-split.md). **Chat / status HTTP helpers** (`post_chat`, async poll): see [`haroma_client.py`](haroma_client.py) and [Training & integrations reference](../docs/reference-training-integrations.md) § HTTP client.

## Quick demo (dry-run)

From the repository root:

```bash
python bridge/sample_http_bridge.py --dry-run
```

## Live demo (Haroma on :8193)

```bash
set HAROMA_URL=http://127.0.0.1:8193
python bridge/sample_http_bridge.py
```

If the server uses **`HAROMA_HTTP_BEARER_TOKEN`**, set the same variable in the bridge process environment; `haroma_client.py` sends `Authorization: Bearer …` automatically for POSTs (and GET `/status` for health checks).

Expect HTTP 200 and JSON containing `agent_environment` echo from the server.

## Integrating ROS 2

1. Create a `rclpy` node on the robot.
2. On each incoming command batch (JSON matching `robot_execution_contract`), run your motion stack.
3. Build feedback with the same fields as `normalize_feedback_payload` (`bridge_schema_version`, `correlation_id`, `results`).
4. Call `post_robot_bridge_feedback(HAROMA_URL, feedback)` — or use your own HTTP client.

Do **not** block high-rate control loops on Haroma; POST feedback asynchronously.

## Python path

Imports use the package name `bridge`. Run scripts from the repo root, or set:

```bash
set PYTHONPATH=C:\path\to\HaromaX6
```

## Tests

See `tests/test_bridge_haroma_client.py` (offline checks for stub + feedback shape).
