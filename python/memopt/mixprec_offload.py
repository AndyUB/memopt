from contextlib import nullcontext
import logging
import torch
from torch.amp import autocast, GradScaler
import torch.nn as nn
import torch.cuda.nvtx as nvtx

from memopt.model.optimizer import AdamW, CPUAdamW
from memopt.model.config import DEFAULT_ADAMW_ARGS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def setup_master_params(model: torch.nn.Module) -> list[torch.Tensor]:
    """
    Create FP32 master copy of parameters.

    Args:
        model: The model whose parameters are to be copied.

    Returns:
        List of FP32 master parameters.
    """
    master_params = []
    for param in model.parameters():
        master_param = param.detach().clone().float()
        master_param.requires_grad = param.requires_grad
        master_params.append(master_param)
    return master_params


class MixedPrecOffloadTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        lower_precision_dtype: torch.dtype | None = None,
        use_cpu_offload: bool = False,
        use_grad_scaler: bool = True,
    ):
        """
        Args:
            model: The model to be trained. Should be initialized in FP32.
            lower_precision_dtype:
                - None or torch.float32: no mixed precision, run everything in FP32.
                - torch.float16 or torch.bfloat16: try to use autocast with this dtype.
            use_cpu_offload: If True, use CPUAdamW so optimizer state lives on CPU.
            use_grad_scaler: If True and mixed precision is enabled on CUDA,
                             use GradScaler for loss scaling.
        """
        self.model = model
        self.use_cpu_offload = use_cpu_offload

        self.device = next(model.parameters()).device

        if lower_precision_dtype is None:
            lower_precision_dtype = torch.float32
        self.lower_precision_dtype = lower_precision_dtype

        self.mixed_precision_enabled = (
            self.lower_precision_dtype != torch.float32 and self.device.type == "cuda"
        )
        if self.lower_precision_dtype != torch.float32 and self.device.type != "cuda":
            logger.warning(
                "Requested lower_precision_dtype=%s but device is %s; "
                "disabling mixed precision.",
                self.lower_precision_dtype,
                self.device.type,
            )

        self.model_dtype = torch.float32
        self.model.to(self.model_dtype)

        if self.mixed_precision_enabled:
            self.forward_context = autocast(
                device_type="cuda",
                dtype=self.lower_precision_dtype,
            )
        else:
            self.forward_context = nullcontext()

        if self.mixed_precision_enabled and use_grad_scaler:
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.optimizer_dtype = torch.float32

        if self.use_cpu_offload and self.device.type == "cuda":
            self.swap_stream = torch.cuda.Stream(device=self.device)
            optimizer_cls = CPUAdamW
        else:
            self.swap_stream = None
            optimizer_cls = AdamW

        params_for_opt = self.model.parameters()

        self.optimizer = optimizer_cls(
            params_for_opt,
            **DEFAULT_ADAMW_ARGS,
            dtype=self.optimizer_dtype,
        )

        self.register_grad_offload_hooks()

    def register_grad_offload_hooks(self):
        """
        Register hooks to offload gradients to CPU after backward,
        if CPU offload is enabled.

        Hooks operate on gradients (Tensors), not parameters:
          new_grad = hook(old_grad)
        """
        if not self.use_cpu_offload or self.device.type != "cuda":
            return

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            def offload_grad_to_cpu(grad, p=param, n=name):
                if grad is None:
                    return None

                if self.swap_stream is not None:
                    with torch.cuda.stream(self.swap_stream):
                        cpu_grad = grad.detach().to("cpu", non_blocking=True)
                else:
                    cpu_grad = grad.detach().to("cpu")

                # Stash CPU grad in optimizer state for this param
                state = self.optimizer.state[p]
                state["cpu_grad"] = cpu_grad

                return grad

            param.register_hook(offload_grad_to_cpu)

    def cast_inputs(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Cast inputs to model dtype (FP32).

        Args:
            inputs: Input tensors.

        Returns:
            Tuple of casted input tensors.
        """
        return tuple(inp.to(self.model_dtype) for inp in inputs)

    def forward(self, *args) -> torch.Tensor:
        """
        Forward pass with autocast if enabled.

        Args:
            args: Input tensors.

        Returns:
            Model output tensor.
        """
        with self.forward_context:
            output = self.model(*args)
        return output

    def backward(self, loss: torch.Tensor) -> None:
        """
        Backward pass.

        - If GradScaler is present, scale(loss).backward().
        - Otherwise, standard loss.backward().
        """
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def step(self, check_overflow: bool = True) -> bool | None:
        """
        Optimizer step with optional overflow checking.

        For AMP + GradScaler:
          - scaler.step(optimizer), scaler.update()
          - No manual overflow logic; GradScaler handles it.

        For pure FP32:
          - Optionally check gradients for NaN/Inf (if check_overflow=True).
          - If overflow detected, skip step and clear grads.

        Returns:
            None if overflow checking is disabled and no scaler is used;
            otherwise, whether the step was considered successful (no overflow).
        """
        if self.scaler is not None:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.zero_grad()
            return True

        no_overflow: bool | None = None
        if check_overflow:
            no_overflow = not self.check_overflow()
            if not no_overflow:
                self.zero_grad()
                logger.warning(
                    "Skipping optimizer step due to gradient overflow. "
                    "Gradients have been cleared."
                )
                return False

        self.optimizer.step()
        self.zero_grad()
        return no_overflow

    def check_overflow(self) -> bool:
        """
        Check if any gradient has inf or nan (for non-AMP path).

        Returns:
            Whether overflow is detected.
        """
        has_overflow = False
        for name, param in self.model.named_parameters():
            if param.grad is not None and not torch.isfinite(param.grad).all():
                has_overflow = True
                logger.warning(f"Overflow detected in gradient of parameter `{name}`")
        return has_overflow

    def zero_grad(self):
        """Zero gradients of model and optimizer."""
        self.model.zero_grad(set_to_none=True)
        self.optimizer.zero_grad(set_to_none=True)


if __name__ == "__main__":
    device = torch.device("cuda")

    class SimpleMLP(nn.Module):
        def __init__(self, in_dim=16, hidden_dim=32, out_dim=8):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, out_dim),
            )

        def forward(self, x):
            return self.net(x)

    def make_data(batch_size=64, in_dim=16, out_dim=8, device="cpu"):
        x = torch.randn(batch_size, in_dim, device=device)
        y = torch.randn(batch_size, out_dim, device=device)
        return x, y

    model = SimpleMLP().to(device)

    trainer_offload = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=None,
        use_cpu_offload=True,
        use_grad_scaler=False,
    )

    trainer = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=None,
        use_cpu_offload=False,
        use_grad_scaler=False,
    )

    print("Running FP32 + CPU offload demo for 5 iterations...")
    x, y = make_data(device=device)

    # warmup
    for step in range(3):
        (x_cast,) = trainer.cast_inputs(x)
        with nvtx.range("forward"):
            out = trainer.forward(x_cast)
        loss = torch.nn.functional.mse_loss(out, y)
        with nvtx.range("backward"):
            trainer.backward(loss)
        with nvtx.range("optimizer_step"):
            ok = trainer.step(check_overflow=True)

    # time 5 iters of non-offload
    import time

    start = time.time()
    for step in range(5):
        (x_cast,) = trainer.cast_inputs(x)
        out = trainer.forward(x_cast)
        loss = torch.nn.functional.mse_loss(out, y)
        trainer.backward(loss)
        ok = trainer.step(check_overflow=True)
    torch.cuda.synchronize()
    end = time.time()
    print(f"FP32 without offload: 5 iters took {end - start:.4f} seconds.")

    # time 5 iters of offload
    start = time.time()
    for step in range(5):
        (x_cast,) = trainer_offload.cast_inputs(x)
        out = trainer_offload.forward(x_cast)
        loss = torch.nn.functional.mse_loss(out, y)
        trainer_offload.backward(loss)
        ok = trainer_offload.step(check_overflow=True)
    torch.cuda.synchronize()
    end = time.time()
    print(f"FP32 with CPU offload: 5 iters took {end - start:.4f} seconds.")
