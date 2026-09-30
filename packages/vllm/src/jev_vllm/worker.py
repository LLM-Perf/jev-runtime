from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class LoRAWorkerExtension:
    """Fixed official worker-extension RPC for a managed LoRA completion fence."""

    # vLLM mixes this extension into its Worker, which provides `device`.
    device: torch.device

    def jev_lora_barrier(self) -> bool:
        import torch

        torch.cuda.synchronize(self.device)
        return True
