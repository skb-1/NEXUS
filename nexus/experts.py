from collections.abc import Callable
import numpy as np
from nexus.router import Routing
from nexus.tensor import Tensor


def dispatch(x: Tensor, routing: Routing, parameter: Callable[[str], Tensor],
             num_experts: int) -> tuple[Tensor, list[int]]:
    result = Tensor(np.zeros_like(x.data), x.backend)
    used = []
    for e in range(num_experts):
        rows, slots = np.nonzero(routing.indices == e)
        if not len(rows):
            continue
        used.append(e)
        # Only selected rows execute this expert. No dense all-expert evaluation.
        y = (x[rows] @ parameter(f"experts.{e}.up")).tanh() @ parameter(f"experts.{e}.down")
        y = y * routing.gates[rows, slots].reshape(-1, 1)
        result = result + y.scatter_rows(rows, len(x.data))
    return result, used
