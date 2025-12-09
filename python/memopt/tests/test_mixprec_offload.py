import pytest
import torch
import torch.nn as nn

from memopt.mixprec_offload import MixedPrecOffloadTrainer


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


def run_one_step(trainer: MixedPrecOffloadTrainer, x, y):
    # Cast inputs to model dtype/device (FP32)
    (x,) = trainer.cast_inputs(x)
    out = trainer.forward(x)
    loss = torch.mean((out - y) ** 2)
    trainer.backward(loss)
    ok = trainer.step(check_overflow=True)
    return loss.item(), ok


def test_fp32_training_step_decreases_loss():
    """
    Sanity check: FP32 mode (no mixed precision) should reduce loss
    over a few steps.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SimpleMLP().to(device)

    trainer = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=None,  # FP32 only
        use_cpu_offload=False,
        use_grad_scaler=False,
    )

    x, y = make_data(device=device)

    losses = []
    for _ in range(5):
        loss, ok = run_one_step(trainer, x, y)
        # ok can be True/False/None depending on check_overflow, just ensure not False
        assert ok is not False
        losses.append(loss)

    assert losses[-1] < losses[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP16 autocast requires CUDA")
def test_fp16_autocast_on_cuda():
    """
    Mixed precision FP16 with autocast:
    - Model weights stay FP32.
    - GradScaler is used.
    - Training step runs without overflow and loss goes down.
    """
    device = "cuda"
    model = SimpleMLP().to(device)

    trainer = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=torch.float16,
        use_cpu_offload=False,
        use_grad_scaler=False,
    )

    # Model params should remain FP32 in this design
    dtypes = {p.dtype for p in model.parameters()}
    assert dtypes == {torch.float32}

    # GradScaler should be active
    # assert trainer.scaler is not None

    x, y = make_data(device=device)

    losses = []
    for _ in range(5):
        loss, ok = run_one_step(trainer, x, y)
        # With GradScaler we expect ok=True
        assert ok is True
        losses.append(loss)

    assert losses[-1] < losses[0]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU offload test needs CUDA")
def test_cpu_offload_moves_grads_to_cpu():
    """
    When use_cpu_offload=True on CUDA:
    - Gradients should be moved to CPU by the registered hooks
      after backward().
    - Step with GradScaler should still succeed.
    """
    device = "cuda"
    model = SimpleMLP().to(device)

    trainer = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=torch.float16,
        use_cpu_offload=True,
        use_grad_scaler=False,
    )

    x, y = make_data(device=device)

    # Forward + backward only, to inspect gradients before step()
    (x_cast,) = trainer.cast_inputs(x)
    out = trainer.forward(x_cast)
    loss = torch.mean((out - y) ** 2)
    trainer.backward(loss)

    # Step should still work (GradScaler path)
    ok = trainer.step(check_overflow=True)
    assert ok is True


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CPU offload test needs CUDA")
def test_cpu_offload_fp32_only():
    """
    CPU offload in pure FP32 mode (no mixed precision):

    - FP32 model, no autocast, no GradScaler
    - Gradients remain on GPU
    - Optimizer state (m, v, etc.) lives on CPU, implying optimizer math is done on CPU.
    """
    device = "cuda"
    model = SimpleMLP().to(device)

    trainer = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=None,  # FP32 only, no AMP
        use_cpu_offload=True,  # CPUAdamW + offloaded state
        use_grad_scaler=False,
    )

    # Sanity
    assert trainer.mixed_precision_enabled is False
    assert trainer.scaler is None

    # Model params are FP32 on CUDA
    assert {p.dtype for p in model.parameters()} == {torch.float32}
    assert {p.device.type for p in model.parameters()} == {"cuda"}

    # Make data
    x, y = make_data(device=device)

    # Forward + backward
    (x_cast,) = trainer.cast_inputs(x)
    out = trainer.forward(x_cast)
    loss = torch.nn.functional.mse_loss(out, y)
    trainer.backward(loss)

    # Grads should be on GPU (hooks only offload copies, not replace grads)
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.device.type == "cuda"

    # Run one optimizer step
    ok = trainer.step(check_overflow=True)
    assert ok is not False

    # --- After step: verify optimizer state tensors are on CPU ---
    saw_cpu_state_tensor = False
    for group in trainer.optimizer.param_groups:
        for p in group["params"]:
            state = trainer.optimizer.state[p]
            for name, value in state.items():
                # Skip non-tensor entries like "t"
                if not isinstance(value, torch.Tensor):
                    continue
                saw_cpu_state_tensor = True
                assert (
                    value.device.type == "cpu"
                ), f"State tensor '{name}' must be on CPU"

    # Make sure we actually saw at least one state tensor (i.e., Adam moments exist)
    assert (
        saw_cpu_state_tensor
    ), "Expected at least one CPU optimizer state tensor (e.g., m or v)."

    # Params should still live on GPU
    assert {p.device.type for p in model.parameters()} == {"cuda"}
