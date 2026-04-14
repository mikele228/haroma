# Step-by-step: connect Haroma to a robot

[<- Back to Index](index.md)

This guide is a **practical sequence** for integrators. The **conceptual contract** (rates, JSON areas, safety layering) is in **[Robot cognitive / control split](robot-cognitive-control-split.md)**. Sample code lives under [`bridge/`](../bridge/README.md).

**Rule:** Haroma is a **cognitive server** (Flask, LLM, threads). **Torque, ESTOP, and hard real-time loops stay on the robot**—never depend on HTTP or the LLM for motor safety.

---

## 1. Decide the responsibility split

| Layer | Robot / RT stack | Haroma |
|-------|------------------|--------|
| Servos, torque @ kHz | Yes | No |
| ESTOP / STO / brakes (physical) | Yes | May only *reflect* state in prompts |
| High-level goals, language, scene | Optional | Yes |
| `body_actions` → command batches | Bridge maps to ROS / drivers | LLM + `build_executor_command_batch` |

---

## 2. Run Haroma where the bridge can reach it

1. Install dependencies (see [Getting Started](getting-started.md)). Optionally run **`python scripts/setup_wizard.py`** and pick **Robot / edge** to set bind address, bearer token, rate limits, and **`HAROMA_ROBOT_CLIENT_URL`**.
2. Launch with `python main.py` (loads `.env` from the project root). Default port is **8193** unless you set **`HAROMA_HTTP_PORT`** (see [`mind/deploy_config.py`](../mind/deploy_config.py)).
3. Put the robot PC (or gateway) and the Haroma host on the same **LAN/VPN**, or run Haroma on-board if resources allow.
4. If you set **`HAROMA_HTTP_BEARER_TOKEN`**, configure the same secret for the bridge so protected POSTs succeed ([`bridge/README.md`](../bridge/README.md)).

---

## 3. Push fused state into cognition (optional, typical)

1. **POST [`/agent/environment`](api-reference.md)** with a JSON snapshot your stack already maintains.
2. Use **`extensions.robot_body`** for embodiment / pose / mode (see [`mind/robot_body_state.py`](../mind/robot_body_state.py) and the tables in [Robot cognitive / control split](robot-cognitive-control-split.md)).
3. **Do not** push raw kHz joint streams through Flask as the only feedback—**close the loop on the robot**; send Haroma a **summary** (often **~5–50 Hz** *downsampled* to something the server can absorb; many deployments use **≤10 Hz** summaries).

---

## 4. Turn cognition into executor command batches

1. Packed LLM output can include **`body_actions`** (see `engine/LLMContextReasoner.py` in-repo).
2. Normalize with **`build_executor_command_batch`** in [`mind/robot_execution_contract.py`](../mind/robot_execution_contract.py) (`bridge_schema_version`, `correlation_id`, `commands[]`).
3. A **bridge process** on the robot (or edge gateway) translates `commands[]` to ROS 2 actions/topics or vendor APIs.
4. **Never** block the motor control loop waiting on the LLM or HTTP.

---

## 5. Send execution feedback back to Haroma

1. After motion / execution, **POST [`/robot/bridge/feedback`](api-reference.md)** with a body matching **`normalize_feedback_payload`** in [`mind/robot_execution_contract.py`](../mind/robot_execution_contract.py) (`command_id`, `status`, optional `t`, etc.).
2. Haroma merges into **`extensions.robot_bridge`**. Inspect **`GET /status`** → **`agent_environment.robot_bridge`** and **`robot_bridge_metrics`** for health.

Keep **`command_id`** and **`correlation_id`** aligned end-to-end for traceability.

---

## 6. ROS 2 wiring (if applicable)

1. Run an **`rclpy`** node on the robot (or a dedicated bridge machine).
2. On each incoming **command batch** (JSON aligned with `robot_execution_contract`), drive your motion stack.
3. Build **`results[]`** for feedback using the same schema as **`normalize_feedback_payload`**.
4. **POST** feedback to Haroma **asynchronously**—do not block high-rate control loops on HTTP.

Optional patterns: [`bridge/ros2_stub.py`](../bridge/ros2_stub.py) (probe/helper, not a full production stack).

---

## 7. Run the reference bridge demo

From the **repository root** (see [`bridge/README.md`](../bridge/README.md)):

```bash
python bridge/sample_http_bridge.py --dry-run
```

With Haroma listening on localhost:

```bash
set HAROMA_URL=http://127.0.0.1:8193
python bridge/sample_http_bridge.py
```

On Linux/macOS, use `export HAROMA_URL=...` instead of `set`.

This exercises HTTP client + stub executor paths before real hardware.

---

## 8. Complete the integration checklist

Work through the checklist in **[Robot cognitive / control split](robot-cognitive-control-split.md#integration-checklist)** (torque only on robot, ESTOP independent of Haroma, bounded env payloads, idempotent `command_id`s, etc.).

---

## Related docs and code

| Topic | Where |
|-------|--------|
| Rates, JSON areas, safety layers | [Robot cognitive / control split](robot-cognitive-control-split.md) |
| HTTP samples, bearer, env vars | [`bridge/README.md`](../bridge/README.md) |
| REST routes | [API Reference](api-reference.md) |
| Command + feedback schema | [`mind/robot_execution_contract.py`](../mind/robot_execution_contract.py) |
| Merge / HTTP bridge helpers | [`integrations/robot_http_bridge.py`](../integrations/robot_http_bridge.py) |
| Environment size / validation | [`mind/environment_context.py`](../mind/environment_context.py) |
