import torch
from typing import Callable

from memopt.model.config import DEFAULT_ADAMW_ARGS
from memopt.model.optimizer import AdamW
from memopt.util import setup_master_params


def make_offload_hook(
    p_cpu: torch.Tensor, copy_stream: torch.cuda.Stream
) -> Callable[[torch.Tensor], None]:
    """
    Hook to offload grad as soon as it's produced on GPU.

    Args:
        p_cpu: Pinned CPU tensor to offload grad into.
        copy_stream: CUDA stream for async copy.

    Returns:
        Hook function.
    """

    def hook(grad: torch.Tensor):
        # grad is on GPU; copy to pinned CPU buffer asynchronously
        with torch.cuda.stream(copy_stream):
            p_cpu.grad.copy_(grad, non_blocking=True)
        # Returning None prevents grad from being stored on GPU
        return None

    return hook


def setup_cpu_offload(
    model: torch.nn.Module, device: torch.device
) -> tuple[list[tuple[torch.nn.Parameter, torch.Tensor]], AdamW, torch.cuda.Stream]:
    """
    Creates a CPU master copy of the model, an optimizer on CPU,
    and sets up:
    - pinned CPU grad buffers
    - pinned CPU param data buffers
    - GPU param hooks for async grad offload

    Args:
        model: The model to offload.
        device: The GPU device to run the model on.

    Returns:
        param_pairs: List of (GPU param, CPU param) pairs.
        optimizer: Optimizer operating on CPU model.
        copy_stream: CUDA stream for async grad copies.
    """
    model.to(device=device)
    copy_stream = torch.cuda.Stream(device=device)

    # Master copy on CPU
    cpu_params = setup_master_params(model, device=torch.device("cpu"), pin_memory=True)
    optimizer = AdamW(cpu_params, **DEFAULT_ADAMW_ARGS)

    param_pairs = list(zip(model.parameters(), cpu_params))
    # Allocate pinned grads and register hooks
    for p_gpu, p_cpu in param_pairs:
        # pinned buffer for grads (CPU, pinned)
        p_cpu.grad = torch.empty_like(p_cpu.data, device="cpu", pin_memory=True)
        p_gpu.register_hook(make_offload_hook(p_cpu, copy_stream))

    return param_pairs, optimizer, copy_stream


class CPUOffloadTrainer:
    def __init__(self, model: torch.nn.Module, device: torch.device):
        self.model = model
        self.device = device

        self.param_pairs, self.optimizer, self.copy_stream = setup_cpu_offload(
            model, device
        )

    def train_step(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> float:
        """
        Single training step:
        - forward/backward on GPU
        - async offload grads to CPU pinned buffers via hooks
        - optimizer step on CPU model
        - async sync of updated CPU weights back to GPU

        Args:
            input: Input tensor.
            target: Target tensor.
            loss_fn: Loss function.

        Returns:
            Loss value.
        """
        input = input.to(self.device)
        target = target.to(self.device)

        # Forward & backward on GPU
        output = self.model(input)
        loss = loss_fn(output, target)
        loss.backward()  # triggers hooks to offload grads GPU -> CPU pinned

        # Ensure all grads are copied to CPU before optimizer step
        self.copy_stream.synchronize()
        self.optimizer.step()

        # Sync updated CPU weights back to GPU, asynchronously
        with torch.no_grad():
            for p_gpu, p_cpu in self.param_pairs:
                with torch.cuda.stream(self.copy_stream):
                    p_gpu.data.copy_(p_cpu.data, non_blocking=True)
        # Wait for copies to finish before next iteration
        self.copy_stream.synchronize()

        return loss.item()


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda", 0)
    model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 10),
    )

    # Setup CPU offload
    trainer = CPUOffloadTrainer(model, device)
    loss_fn = torch.nn.CrossEntropyLoss()
    # Dummy batch
    batch_size = 32
    x = torch.randn(batch_size, 128)
    y = torch.randint(0, 10, (batch_size,))

    num_steps = 5
    for _ in range(num_steps):
        loss_val = trainer.train_step(
            x,
            y,
            loss_fn,
        )
        print("Loss:", loss_val)
