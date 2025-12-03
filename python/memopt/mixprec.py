from contextlib import nullcontext
import logging
import torch

from memopt.model.optimizer import AdamW
from memopt.model.config import DEFAULT_ADAMW_ARGS

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def setup_master_params(
    model: torch.nn.Module, device: torch.device
) -> list[torch.Tensor]:
    """
    Create FP32 master copy of parameters.

    Args:
        model: The model whose parameters are to be copied.
        device: The device to place the master parameters on.

    Returns:
        List of FP32 master parameters.
    """
    master_params = []
    for param in model.parameters():
        master_param = param.detach().clone().float().to(device)
        master_param.requires_grad = param.requires_grad
        master_params.append(master_param)
    return master_params


def assert_model_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """
    Assert that all model parameters are of the specified dtype.

    Args:
        model: The model to check.
        dtype: The expected data type.
    """
    for param in model.parameters():
        if param.dtype != dtype:
            raise ValueError(
                f"Model parameter has dtype {param.dtype}, expected {dtype}."
            )


class MixedPrecisionTrainer:
    """
    Mixed precision training manager.
    - Lower precision (FP16/BF16) for forward and backward arithmetic.
    - Parameters cast to lower precision for forward computation.
    - Activations and gradients in lower precision.
    - (Optional) Master copy of parameters in FP32 for optimizer updates.
    - If master copy is used, optimizer states are in FP32; otherwise,
      they are in lower precision.
    - (Optional) Loss scaling to prevent gradient underflow (for FP16).
    """

    def __init__(
        self,
        model: torch.nn.Module,
        lower_precision_dtype: torch.dtype,
        device: torch.device,
        use_master_copy: bool = True,
        loss_scale: float | None = None,
        use_autocast: bool = False,
    ):
        """
        Args:
            model: The model to be trained. Should be initialized in FP32.
            lower_precision_dtype: The lower precision dtype.
            device: The device to run the model on.
            use_master_copy: Whether to maintain FP32 master copy of parameters.
            loss_scale: Constant loss scale factor. If None, loss is not scaled.
            use_autocast: Whether to use torch.autocast for automatic casting.
        """
        if loss_scale is not None and lower_precision_dtype == torch.bfloat16:
            logger.warning("Loss scaling is typically not needed for BF16.")

        self.model = model
        self.lower_precision_dtype = lower_precision_dtype
        self.device = device
        self.use_master_copy = use_master_copy
        self.loss_scale = loss_scale
        self.use_autocast = use_autocast

        self.model_dtype = lower_precision_dtype
        self.master_params: list[torch.Tensor] | None = None
        if use_master_copy:
            if use_autocast:
                # If using autocast, master params are not needed as autocast
                # handles casting internally. Keep model in FP32.
                self.model_dtype = torch.float32
            else:
                self.master_params = setup_master_params(model, device)
        model.to(device=device, dtype=self.model_dtype)

        self.optimizer_dtype = (
            torch.float32 if use_master_copy else lower_precision_dtype
        )
        self.optimizer = AdamW(
            (
                self.master_params
                if self.master_params is not None
                else model.parameters()
            ),
            **DEFAULT_ADAMW_ARGS,
            dtype=self.optimizer_dtype,
        )

    def cast_inputs(self, *inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Cast inputs to model dtype.

        Args:
            inputs: Input tensors.

        Returns:
            Tuple of casted input tensors.
        """
        casted_inputs = tuple(
            inp.to(device=self.device, dtype=self.model_dtype) for inp in inputs
        )
        return casted_inputs

    def forward_context(self) -> torch.Tensor:
        """
        Context for forward pass in lower precision.
        """
        if self.use_autocast:
            return torch.autocast(
                device_type=self.device.type,
                dtype=self.lower_precision_dtype,
            )
        else:
            return nullcontext()

    def scale_loss(self, loss: torch.Tensor) -> torch.Tensor:
        """
        Scale loss to prevent gradient underflow, if loss scaling is enabled.

        Args:
            loss: Loss tensor.

        Returns:
            Scaled loss tensor.
        """
        if self.loss_scale is not None:
            scaled_loss = loss * self.loss_scale
        else:
            scaled_loss = loss
        return scaled_loss

    def step(self, check_overflow: bool = True) -> bool | None:
        """
        Optimizer step with gradient unscaling and optional overflow checking.

        Args:
            check_overflow: Whether to check for gradient overflow.

        Returns:
            None if overflow checking is disabled; otherwise, whether the step
            was successful (no overflow).
        """
        # Check for inf/nan in gradients (overflow detection)
        no_overflow: bool | None = None
        if check_overflow:
            no_overflow = not self.check_overflow()

            if not no_overflow:
                # Skip optimizer step
                self.zero_grad()
                logger.warning(
                    "Skipping optimizer step due to gradient overflow. "
                    "Gradients have been cleared."
                )
                return False

        self.unscale_and_copy_gradients()
        self.optimizer.step()
        self.copy_master_to_model()
        self.zero_grad()

        return no_overflow

    def check_overflow(self) -> bool:
        """
        Check if any gradient has inf or nan.

        Returns:
            Whether overflow is detected.
        """
        has_overflow = False
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if not torch.isfinite(param.grad).all():
                    has_overflow = True
                    logger.warning(
                        f"Overflow detected in gradient of parameter `{name}`: "
                        f"{param.grad}"
                    )
        return has_overflow

    @torch.no_grad()
    def unscale_and_copy_gradients(self):
        """
        Unscale gradients if loss scaling is enabled, and copy from model
        (lower precision) to master params (FP32) if master params are maintained.
        """
        if self.master_params is None:
            if self.loss_scale is None:
                return

            for model_param in self.model.parameters():
                if model_param.grad is not None:
                    model_param.grad /= self.loss_scale
            return

        model_params = list(self.model.parameters())
        for model_param, master_param in zip(model_params, self.master_params):
            unscaled_grad = None
            if model_param.grad is not None:
                if self.loss_scale is not None:
                    unscaled_grad = model_param.grad.float() / self.loss_scale
                else:
                    unscaled_grad = model_param.grad.float()
            master_param.grad = unscaled_grad

    @torch.no_grad()
    def copy_master_to_model(self):
        """
        Copy updated master parameters back to model parameters,
        if master params are maintained.
        """
        if self.master_params is None:
            return

        model_params = list(self.model.parameters())
        for model_param, master_param in zip(model_params, self.master_params):
            model_param.copy_(master_param.to(self.lower_precision_dtype))

    def zero_grad(self):
        """Zero gradients of model and optimizer."""
        self.model.zero_grad()
        self.optimizer.zero_grad()
