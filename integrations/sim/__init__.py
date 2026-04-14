"""Universal simulation integration — pluggable backends for any simulator host.

Use :func:`load_backend_from_env` after setting ``HAROMA_SIM_BACKEND`` (see docs).

Built-in names: ``null``, ``http_json``. Any other engine: ``mypkg.my_sim:BackendClass``
with optional ``HAROMA_SIM_BACKEND_KWARGS`` JSON.
"""

from integrations.sim.protocol import SimulationBackend
from integrations.sim.registry import (
    create_backend,
    load_backend_from_env,
    register_backend,
)
from integrations.sim.universal import (
    merge_simulation_into_extensions,
    simulation_summary_for_prompt,
)

__all__ = [
    "SimulationBackend",
    "create_backend",
    "load_backend_from_env",
    "register_backend",
    "merge_simulation_into_extensions",
    "simulation_summary_for_prompt",
]
