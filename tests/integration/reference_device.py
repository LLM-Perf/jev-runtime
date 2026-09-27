"""Bounded CUDA execution for independent, non-serving numerical references.

Only module-owned leaf state is supported. This is intentionally separate from
the runtime and does not implement model offloading in either serving engine.
"""

from __future__ import annotations


def cuda_budget(free: int, total: int, budget_mib: int, reserve_mib: int) -> int:
    """Return a capped allocator budget without consuming the requested reserve."""
    if not 0 < free <= total or budget_mib <= 0 or reserve_mib < 3072:
        raise ValueError("Reference requires positive memory and at least a 3072 MiB reserve")
    available = free - reserve_mib * 1024**2
    if available <= 0:
        raise ValueError("Insufficient free GPU memory outside the reference reserve")
    return min(budget_mib * 1024**2, available)


class CudaLeafOffload:
    """Move leaf-owned parameters/buffers around each module's CUDA forward.

    The caller supplies CUDA inputs, disables KV caching, and installs any output
    projection hooks *before* entering. Ancestor modules reading child weights
    directly are unsupported. Unexpected CPU/CUDA access fails the forward; this
    helper never retries with a different computation or reports a failed run as
    a numerical pass. Validate resident/offload equality for every new model family.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.modules = []
        self.visited = set()

    def __enter__(self):
        # Validate the complete model before registering hooks or moving state.
        for name, module in self.model.named_modules():
            state = list(module.parameters(recurse=False)) + list(module.buffers(recurse=False))
            if not state:
                continue
            if any(
                list(child.parameters()) or list(child.buffers()) for child in module.children()
            ):
                raise ValueError(f"Offload requires leaf-owned state: {name}")
            if any(t.device.type != "cpu" for t in state):
                raise ValueError(f"Offload requires initial CPU state: {name}")
            self.modules.append((name, module))
        for name, module in self.modules:

            def upload(module, inputs, name=name):
                module.to("cuda")
                self.visited.add(name)

            def download(module, inputs, output):
                module.to("cpu")

            self.handles.append(module.register_forward_pre_hook(upload))
            self.handles.append(module.register_forward_hook(download, always_call=True))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.model.to("cpu")

    def metadata(self):
        return {
            "strategy": "module_owned_leaf_state",
            "registered_modules": [name for name, _ in self.modules],
            "visited_modules": sorted(self.visited),
            "unvisited_modules": sorted({name for name, _ in self.modules} - self.visited),
        }
