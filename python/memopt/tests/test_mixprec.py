import itertools
import logging
import pytest
import torch
import torch.nn as nn

from memopt.util import set_seed
from memopt.mixprec import MixedPrecisionTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


@pytest.mark.parametrize("lower_precision_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("use_master_copy", [True, False])
@pytest.mark.parametrize("loss_scale", [None, 8])
@pytest.mark.parametrize("use_autocast", [True, False])
@pytest.mark.parametrize("device", ["cuda", "cpu"])
def test_sequential_model(
    lower_precision_dtype: torch.dtype,
    use_master_copy: bool,
    loss_scale: float | None,
    use_autocast: bool,
    device: str,
) -> None:
    """
    Test mixed precision training on a simple sequential model with all possible
    configurations.
    """
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is not available.")
    device = torch.device(device)

    set_seed()
    model = nn.Sequential(
        nn.Linear(10, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 1),
    )

    mp_trainer = MixedPrecisionTrainer(
        model=model,
        lower_precision_dtype=lower_precision_dtype,
        device=device,
        use_master_copy=use_master_copy,
        loss_scale=loss_scale,
        use_autocast=use_autocast,
    )

    set_seed()
    x = torch.randn(32, 10)
    y = torch.randn(32, 1)

    for step in range(5):
        x, y = mp_trainer.cast_inputs(x, y)
        with mp_trainer.forward_context():
            output = model(x)
            loss = torch.mean((output - y) ** 2)
        scaled_loss = mp_trainer.scale_loss(loss)
        scaled_loss.backward()
        success = mp_trainer.step()

        logger.info(f"Step {step}, Loss: {loss.item()}, Step success: {success}")


if __name__ == "__main__":
    args_combinations = itertools.product(
        [torch.float16, torch.bfloat16],
        [True, False],
        [None, 8],
        [True, False],
        ["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"],
    )
    for dtype, use_master, scale, autocast, device in args_combinations:
        logger.info(
            f"=== dtype={dtype}, use_master_copy={use_master}, "
            f"loss_scale={scale}, use_autocast={autocast}, device={device} ==="
        )
        test_sequential_model(
            lower_precision_dtype=dtype,
            use_master_copy=use_master,
            loss_scale=scale,
            use_autocast=autocast,
            device=device,
        )
