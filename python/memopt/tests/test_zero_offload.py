import os
import time
import torch
from typing import Callable

from memopt.model.mlp import MLP
from memopt.model.transformer import Transformer
from memopt.model.optimizer import cross_entropy_loss
from memopt.model.config import (
    DEFAULT_ADAMW_ARGS,
    MLP_TINY as MLP_ARGS,
    TRANSFORMER_SMALL as TRANSFORMER_ARGS,
)
from memopt.zero_offload import ZeroOffloadTrainer
from memopt.util import set_seed, get_timing_events, measure_peak_memory, MemoryStats

CHECK_OVERFLOW = os.environ.get("CHECK_OVERFLOW", "0") == "1"


def print_zero_offload_results(
    offload: bool,
    dtype: torch.dtype,
    elapse_ms: float,
    mem_base: MemoryStats,
    mem_final: MemoryStats,
):
    label = f"Offload (dtype={dtype})" if offload else "No offload"
    print(f"========== {label} ==========")
    print(f"Time elapsed: {elapse_ms:.2f} ms")
    print(f"Memory usage (base): {mem_base}")
    print(f"Memory usage (final): {mem_final}")


def zero_offload_correctness(
    get_model_fn: Callable[[torch.device], torch.nn.Module],
    get_data_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    dtype: torch.dtype,
    loss_scale: float | None = None,
    param_atol: float = 1e-6,
    loss_atol: float = 1e-6,
):
    assert torch.cuda.device_count() >= 2, "Need at least 2 CUDA devices for this test."

    state_dict = get_model_fn(torch.device("cpu")).state_dict()
    device = torch.device("cuda", 0)
    device_offload = torch.device("cuda", 1)

    model = get_model_fn(device)
    model.load_state_dict(state_dict)
    optimizer = torch.optim.AdamW(model.parameters(), **DEFAULT_ADAMW_ARGS)
    model_offload = get_model_fn(device_offload)
    model_offload.load_state_dict(state_dict)
    trainer = ZeroOffloadTrainer(
        model_offload,
        dtype=dtype,
        device=device_offload,
        loss_scale=loss_scale,
    )

    set_seed()
    input, target = get_data_fn()

    # No offload
    x = input.to(device)
    y = target.to(device)

    mem_base = measure_peak_memory(device)
    start_ev, end_ev = get_timing_events(num_events=2)
    torch.cuda.synchronize(device)
    start_ev.record()
    out = model(x)
    loss = loss_fn(out, y)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    end_ev.record()
    torch.cuda.synchronize(device)
    elapse_ms = start_ev.elapsed_time(end_ev)
    mem_final = measure_peak_memory(device)
    loss_val = loss.item()

    print_zero_offload_results(False, torch.float32, elapse_ms, mem_base, mem_final)

    # With offload
    x_offload = input.to(device_offload)
    y_offload = target.to(device_offload)

    mem_base_offload = measure_peak_memory(device_offload)
    start_ev_offload, end_ev_offload = get_timing_events(num_events=2)
    torch.cuda.synchronize(device_offload)
    start_ev_offload.record()
    start_time = time.perf_counter()
    ok, loss_val_offload = trainer.train_iter(
        x_offload, y_offload, loss_fn, check_overflow=CHECK_OVERFLOW
    )
    end_time = time.perf_counter()
    end_ev_offload.record()
    torch.cuda.synchronize(device_offload)
    elapse_ms_offload_wall = (end_time - start_time) * 1000
    elapse_ms_offload = start_ev_offload.elapsed_time(end_ev_offload)
    mem_final_offload = measure_peak_memory(device_offload)

    print_zero_offload_results(
        True, dtype, elapse_ms_offload, mem_base_offload, mem_final_offload
    )
    print(
        f"With offload: wall time = {elapse_ms_offload_wall:.2f} ms, "
        f"diff = {elapse_ms_offload_wall - elapse_ms_offload:.2f} ms"
    )
    print(f"Loss (no offload): {loss_val}")
    print(f"Loss (with offload): {loss_val_offload}")
    print(f"Offload step successful: {ok}")

    # Compare memory usage
    assert mem_final_offload.max_allocated_bytes <= mem_final.max_allocated_bytes
    assert mem_final_offload.curr_allocated_bytes < mem_final.curr_allocated_bytes

    # Compare losses
    assert torch.isclose(
        torch.tensor(loss_val, device="cpu"),
        torch.tensor(loss_val_offload, device="cpu"),
        atol=loss_atol,
    ), f"Loss mismatch: {loss_val} vs {loss_val_offload}"

    if dtype != torch.float32:
        print("Skipping parameter correctness check for non-FP32 dtype.")
        return
    if not CHECK_OVERFLOW:
        print("Skipping parameter correctness check since overflow check is disabled.")
        return
    if not ok:
        print("Offload step failed; skipping parameter correctness check.")
        return

    # Compare model parameters
    for p1, p2 in zip(model.parameters(), model_offload.parameters()):
        p1 = p1.detach().cpu()
        p2 = p2.detach().cpu()
        max_diff = torch.max(torch.abs(p1 - p2)).item()
        assert torch.allclose(
            p1, p2, atol=param_atol
        ), f"Model parameters mismatch: max_diff = {max_diff}, {p1} vs {p2}"


def test_mlp_offload_correctness(dtype: torch.dtype, loss_scale: float | None):
    extra_args = {}
    if dtype == torch.float16 or dtype == torch.bfloat16:
        extra_args["loss_atol"] = 1e-2
    zero_offload_correctness(
        get_model_fn=lambda device: MLP(device=device, **MLP_ARGS),
        get_data_fn=lambda: (
            torch.randn(32, MLP_ARGS["input_dim"]),
            torch.randn(32, MLP_ARGS["output_dim"]),
        ),
        dtype=dtype,
        loss_scale=loss_scale,
        loss_fn=lambda out, target: torch.mean((out - target) ** 2),
        param_atol=2e-5,
        **extra_args,
    )


def gen_transformer_batch():
    batch_size = 8
    seq_length = 16
    vocab_size = TRANSFORMER_ARGS["vocab_size"]
    batch = torch.randint(
        low=0,
        high=vocab_size,
        size=(batch_size, seq_length + 1),
        device="cpu",
    )
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]
    return input_ids, target_ids


def test_transformer_offload_correctness(dtype: torch.dtype, loss_scale: float | None):
    extra_args = {}
    if dtype == torch.float16 or dtype == torch.bfloat16:
        extra_args["loss_atol"] = 1e-1
    zero_offload_correctness(
        get_model_fn=lambda device: Transformer(device=device, **TRANSFORMER_ARGS),
        get_data_fn=gen_transformer_batch,
        loss_fn=cross_entropy_loss,
        dtype=dtype,
        loss_scale=loss_scale,
        param_atol=2e-3,
        **extra_args,
    )


if __name__ == "__main__":
    args_combinations = [
        (torch.float32, None),
        (torch.float16, 8192),
        (torch.float16, None),
        (torch.bfloat16, 8192),
        (torch.bfloat16, None),
    ]
    for dtype, loss_scale in args_combinations:
        print(
            f"=== Testing zero offload correctness: dtype={dtype}, "
            f"loss_scale={loss_scale} ==="
        )
        test_mlp_offload_correctness(dtype, loss_scale)
        test_transformer_offload_correctness(dtype, loss_scale)
