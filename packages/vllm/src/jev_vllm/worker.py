class LoRAWorkerExtension:
    """Fixed official worker-extension RPC for a managed LoRA completion fence."""

    def jev_lora_barrier(self) -> bool:
        import torch

        torch.cuda.synchronize(self.device)
        return True
