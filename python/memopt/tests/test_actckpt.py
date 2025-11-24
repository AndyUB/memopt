import torch
from typing import Any, Mapping

from memopt.actckpt import checkpoint
from memopt.model.mlp import MLP
from memopt.model.optimizer import DEFAULT_ADAMW_ARGS, AdamW
from memopt.util import (
    MemoryStats,
    measure_peak_memory,
    set_seed,
    start_memory_tracing,
    stop_memory_tracing,
)

INPUT_DIM = 10
HIDDEN_DIM = 20
OUTPUT_DIM = 5


def train_mlp(
    model_states: Mapping[str, Any],
    input: torch.Tensor,
    device: torch.device,
    use_checkpoint: bool,
) -> dict[str, Any]:
    start_memory_tracing()

    model = MLP(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, device=device).to(device)
    model.load_state_dict(model_states)
    optimizer = AdamW(model.parameters(), **DEFAULT_ADAMW_ARGS)
    input = input.to(device)

    pre_fwd_mem = measure_peak_memory(device)
    if use_checkpoint:
        input.requires_grad_()
        output = checkpoint(model, input)
    else:
        output = model(input)
    loss = torch.sum(output)
    post_fwd_mem = measure_peak_memory(device)

    loss.backward()
    input.grad = None
    optimizer.step()

    stop_memory_tracing("ckpt.pickle" if use_checkpoint else "no_ckpt.pickle")

    output = output.detach().cpu()
    model_state = {
        name: param.detach().cpu() for name, param in model.named_parameters()
    }
    grads = {
        name: param.grad.detach().cpu() for name, param in model.named_parameters()
    }
    return {
        "output": output,
        "model_state": model_state,
        "grads": grads,
        "pre_fwd_mem": pre_fwd_mem,
        "post_fwd_mem": post_fwd_mem,
    }


def test_mlp_actckpt():
    assert torch.cuda.device_count() >= 2, "This test requires at least 2 CUDA devices."
    set_seed()

    batch_size = 4
    input = torch.randn(batch_size, INPUT_DIM)
    state_dict = MLP(
        INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, device=torch.device("cpu")
    ).state_dict()

    result = train_mlp(state_dict, input, torch.device("cuda:0"), False)
    result_ckpt = train_mlp(state_dict, input, torch.device("cuda:1"), True)

    assert torch.allclose(
        result["output"], result_ckpt["output"]
    ), "Outputs do not match!"
    for name in result["grads"]:
        assert torch.allclose(
            result["grads"][name], result_ckpt["grads"][name]
        ), f"Gradients for {name} do not match!"
    for name in result["model_state"]:
        assert torch.allclose(
            result["model_state"][name], result_ckpt["model_state"][name]
        ), f"Model parameters for {name} do not match!"

    post_fwd_mem: MemoryStats = result["post_fwd_mem"]
    post_fwd_mem_ckpt: MemoryStats = result_ckpt["post_fwd_mem"]
    assert (
        post_fwd_mem_ckpt.max_allocated_bytes <= post_fwd_mem.max_allocated_bytes
    ), "Checkpointing increased peak memory usage during forward pass!"
    assert (
        post_fwd_mem_ckpt.curr_allocated_bytes < post_fwd_mem.curr_allocated_bytes
    ), "Checkpointing did not reduce final memory usage after forward pass!"
    pre_fwd_mem: MemoryStats = result["pre_fwd_mem"]
    pre_fwd_mem_ckpt: MemoryStats = result_ckpt["pre_fwd_mem"]
    print(f"[Info] No checkpointing - pre-forward memory: {pre_fwd_mem}")
    print(f"[Info] Checkpointing - pre-forward memory: {pre_fwd_mem_ckpt}")
    print(f"[Info] No checkpointing - post-forward memory: {post_fwd_mem}")
    print(f"[Info] Checkpointing - post-forward memory: {post_fwd_mem_ckpt}")


if __name__ == "__main__":
    test_mlp_actckpt()
