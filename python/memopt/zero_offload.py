import logging
import torch
from typing import Callable

from memopt.model.optimizer import AdamW
from memopt.model.config import DEFAULT_ADAMW_ARGS
from memopt.util import assert_model_dtype, setup_master_params


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


class MasterParam:
    def __init__(
        self,
        master_param: torch.Tensor,
        device_dtype: torch.dtype,
        copy_stream: torch.cuda.Stream,
    ):
        """
        Args:
            master_param: The parameter stored on CPU in FP32.
            device_dtype: The dtype of the parameter on device.
            copy_stream: CUDA stream for async copies.
        """
        if master_param.dtype != torch.float32:
            raise ValueError("Master parameter must be in FP32.")
        if master_param.device.type != "cpu":
            raise ValueError("Master parameter must be on CPU.")

        self.device_dtype = device_dtype
        self.copy_stream = copy_stream
        self.casting, self.master_param, self.param_buf, self.grad_buf = (
            self._get_param_grad_bufs(master_param, device_dtype)
        )

    def _get_param_grad_bufs(
        self, master_param: torch.Tensor, device_dtype: torch.dtype
    ) -> tuple[bool, torch.Tensor, torch.Tensor, torch.Tensor]:
        casting = device_dtype != torch.float32
        if casting:
            param_buf = torch.empty_like(
                master_param, device="cpu", pin_memory=True, dtype=device_dtype
            )
            grad_buf = torch.empty_like(param_buf, pin_memory=True)
            master_param.grad = None
        else:
            param_buf = master_param.pin_memory()
            master_param = param_buf
            grad_buf = torch.empty_like(master_param, pin_memory=True)
            master_param.grad = grad_buf
        return casting, master_param, param_buf, grad_buf

    def copy_param(self, device_param: torch.Tensor) -> None:
        """
        Copy parameter from CPU master to device parameter.
        """
        if self.casting:
            self.param_buf.copy_(self.master_param.to(dtype=self.device_dtype))
        with torch.cuda.stream(self.copy_stream):
            device_param.copy_(self.param_buf, non_blocking=True)

    def copy_grad(self, device_grad: torch.Tensor) -> None:
        """
        Copy gradient from device to CPU grad buffer.
        """
        with torch.cuda.stream(self.copy_stream):
            self.grad_buf.copy_(device_grad, non_blocking=True)

    def cast_grad(self) -> None:
        """
        Cast gradient to FP32 and store in master parameter grad.
        """
        if self.casting:
            self.master_param.grad = self.grad_buf.to(dtype=torch.float32)


def make_zero_offload_hook(master_param: MasterParam) -> Callable[[torch.Tensor], None]:
    """
    Zero offload hook to offload grad into CPU master param.

    Args:
        master_param: Master parameter.

    Returns:
        Hook function.
    """

    def hook(grad: torch.Tensor):
        # grad is on GPU; copy to pinned CPU buffer asynchronously
        master_param.copy_grad(grad)
        # Returning None prevents grad from being stored on GPU
        return None

    return hook


def setup_zero_offload(
    model: torch.nn.Module, device: torch.device, dtype: torch.dtype
) -> tuple[list[MasterParam], AdamW, torch.cuda.Stream]:
    """
    Creates a CPU master copy of the model, an optimizer on CPU, and sets up:
    - pinned CPU grad buffers
    - pinned CPU param data buffers
    - GPU param hooks for async grad offload

    Args:
        model: The model to offload.
        device: The GPU device to run the model on.
        dtype: The dtype to use for the model on device.

    Returns:
        master_params: Master parameters on CPU.
        optimizer: Optimizer operating on CPU model.
        copy_stream: CUDA stream for async grad copies.
    """
    copy_stream = torch.cuda.Stream(device=device)

    # Master copy on CPU
    cpu_params = setup_master_params(model, device=torch.device("cpu"), pin_memory=True)
    optimizer = AdamW(cpu_params, **DEFAULT_ADAMW_ARGS)

    # Cast model to device dtype
    model.to(device=device, dtype=dtype)

    master_params = [MasterParam(p, dtype, copy_stream) for p in cpu_params]
    param_pairs = list(zip(model.parameters(), master_params))
    # Allocate pinned grads and register hooks
    for p_gpu, p_master in param_pairs:
        p_gpu.register_hook(make_zero_offload_hook(p_master))

    return master_params, optimizer, copy_stream


class ZeroOffloadTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        dtype: torch.dtype,
        device: torch.device,
        loss_scale: float | None = None,
    ):
        """
        Args:
            model: The model to be trained. Should be initialized in FP32.
            dtype: The dtype of the model on device.
            device: The device to run the model on.
            loss_scale: Constant loss scale factor. If None, loss is not scaled.
        """
        assert_model_dtype(model, torch.float32)
        if loss_scale is not None and dtype == torch.bfloat16:
            logger.warning("Loss scaling is typically not needed for BF16.")

        self.model = model
        self.dtype = dtype
        self.device = device
        self.loss_scale = loss_scale

        self.master_params, self.optimizer, self.copy_stream = setup_zero_offload(
            model, device, dtype
        )

    def cast_inputs(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Cast inputs to model dtype.
        """
        return tuple(inp.to(device=self.device, dtype=self.dtype) for inp in inputs)

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss by loss_scale if set.
        """
        if self.loss_scale is not None:
            return loss * self.loss_scale
        return loss

    def step(self, check_overflow: bool = True) -> bool | None:
        """
        Perform optimizer step with zero offload.

        Args:
            check_overflow: If True, check for overflow in gradients.

        Returns:
            If check_overflow is True, whether overflow occurred.
            Otherwise, None.
        """
        # Wait for all async copies to complete
        self.copy_stream.synchronize()
        for master_param in self.master_params:
            master_param.cast_grad()

        result_ok: bool | None = None
        if check_overflow:
            overflow = self.check_overflow()
            if overflow:
                logger.warning("Overflow detected. Skipping step.")
                return False
            result_ok = True

        self.optimizer.step()
        # Copy updated params back to device
        with torch.no_grad():
            for p_gpu, master_param in zip(self.model.parameters(), self.master_params):
                master_param.copy_param(p_gpu)
        self.copy_stream.synchronize()

        return result_ok

    def check_overflow(self) -> bool:
        """
        Check if any gradient has inf or nan.

        Returns:
            Whether overflow is detected.
        """
        for param in self.master_params:
            if not torch.isfinite(param.grad_buf).all():
                return True
        return False

    def train_iter(
        self,
        input: torch.Tensor,
        target: torch.Tensor,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        check_overflow: bool = True,
    ) -> tuple[bool | None, float]:
        """
        Perform a training iteration.

        Args:
            input: Input tensor.
            target: Target tensor.
            loss_fn: Loss function.
            check_overflow: Whether to check for overflow during step.

        Returns:
            Whether the step was successful (no overflow if checked), and loss value.
        """
        input, target = self.cast_inputs(input, target)
        output = self.model(input)
        loss = loss_fn(output, target)
        scaled_loss = self.scale_loss(loss)
        scaled_loss.backward()
        ok = self.step(check_overflow=check_overflow)
        return ok, loss.item()
