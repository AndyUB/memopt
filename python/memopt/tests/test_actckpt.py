import logging
import pytest
import torch
from typing import Any, Mapping, Type

from memopt.actckpt import checkpoint
from memopt.model.config import (
    DEFAULT_ADAMW_ARGS,
    MLP_TINY as MLP_ARGS,
    TRANSFORMER_DEFAULT_BATCH_SIZE,
    TRANSFORMER_LARGE,
    TRANSFORMER_LARGE_SINGLE_PROCESS_CONTEXT_LENGTH,
)
from memopt.model.mlp import MLP
from memopt.model.optimizer import AdamW, cross_entropy_loss
from memopt.model.ckpt_transformer import (
    AttnCheckpointedTransformer,
    BlockwiseCheckpointedTransformer,
    FFNCheckpointedTransformer,
)
from memopt.model.transformer import Transformer
from memopt.util import (
    MemoryStats,
    compute_elapses_stats,
    log_elapses,
    measure_peak_memory,
    set_seed,
    start_memory_tracing,
    stop_memory_tracing,
    TrainingTimer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def train_mlp_ckpt(
    model_states: Mapping[str, Any],
    input: torch.Tensor,
    device: torch.device,
    use_checkpoint: bool,
) -> dict[str, Any]:
    start_memory_tracing()

    model = MLP(device=device, **MLP_ARGS).to(device)
    model.load_state_dict(model_states)
    optimizer = AdamW(model.parameters(), **DEFAULT_ADAMW_ARGS)
    input = input.to(device)

    pre_fwd_mem = measure_peak_memory(device)
    if use_checkpoint:
        output = checkpoint(model, input)
    else:
        output = model(input)
    loss = torch.sum(output)
    post_fwd_mem = measure_peak_memory(device)

    loss.backward()
    optimizer.step()

    stop_memory_tracing("mlp_ckpt.pickle" if use_checkpoint else "mlp_nockpt.pickle")

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
    logger.info("=== Test MLP ===")
    logger.info(
        f"Test MLP activation checkpointing: args={MLP_ARGS}, batch_size={batch_size}"
    )
    input = torch.randn(batch_size, MLP_ARGS["input_dim"])
    state_dict = MLP(device=torch.device("cpu"), **MLP_ARGS).state_dict()

    result = train_mlp_ckpt(state_dict, input, torch.device("cuda:0"), False)
    result_ckpt = train_mlp_ckpt(state_dict, input, torch.device("cuda:1"), True)

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
    logger.info(f"(MLP) No checkpointing - pre-forward memory: {pre_fwd_mem}")
    logger.info(f"(MLP) Checkpointing - pre-forward memory: {pre_fwd_mem_ckpt}")
    logger.info(f"(MLP) No checkpointing - post-forward memory: {post_fwd_mem}")
    logger.info(f"(MLP) Checkpointing - post-forward memory: {post_fwd_mem_ckpt}")


def profile_transformer_ckpt_memory(
    model_states: Mapping[str, Any],
    batch: torch.Tensor,
    device: torch.device,
    transformer_cls: Type[Transformer],
    ckpt_filename: str,
) -> dict[str, Any]:
    start_memory_tracing()

    model = transformer_cls(
        device=device,
        **TRANSFORMER_LARGE,
    ).to(device)
    model.load_state_dict(model_states)
    optimizer = AdamW(model.parameters(), **DEFAULT_ADAMW_ARGS)
    batch = batch.to(device)
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]

    pre_fwd_mem = measure_peak_memory(device)
    output = model(input_ids)
    loss = cross_entropy_loss(output, target_ids)
    post_fwd_mem = measure_peak_memory(device)

    loss.backward()
    optimizer.step()

    stop_memory_tracing(ckpt_filename)

    model_state = {
        name: param.detach().cpu() for name, param in model.named_parameters()
    }
    return {
        "model_state": model_state,
        "pre_fwd_mem": pre_fwd_mem,
        "post_fwd_mem": post_fwd_mem,
    }


def time_transformer_ckpt(
    batch: torch.Tensor,
    device: torch.device,
    transformer_cls: Type[Transformer],
) -> dict[str, Any]:
    batch_size, sample_len = batch.shape
    seq_len = sample_len - 1
    cls_name = transformer_cls.__name__
    logger.info(f"=== Time {cls_name} ===")
    logger.info(
        f"Time Transformer activation checkpointing: args={TRANSFORMER_LARGE}, "
        f"batch_size={batch_size}, seq_len={seq_len}, transformer_cls={cls_name}"
    )

    set_seed()
    model = transformer_cls(
        device=device,
        **TRANSFORMER_LARGE,
    ).to(device)
    optimizer = AdamW(model.parameters(), **DEFAULT_ADAMW_ARGS)

    batch = batch.to(device)
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]

    num_warmup_iters = 5
    num_benchmark_iters = 10
    elapses_across_iters: list[dict[str, float]] = []
    for _ in range(num_warmup_iters + num_benchmark_iters):
        timer = TrainingTimer()

        torch.cuda.synchronize(device)
        timer.start_event.record()
        optimizer.zero_grad()

        timer.forward_start_event.record()
        output = model(input_ids)
        timer.forward_end_event.record()
        loss = cross_entropy_loss(output, target_ids)

        timer.backward_start_event.record()
        loss.backward()
        timer.backward_end_event.record()
        optimizer.step()
        timer.end_event.record()
        torch.cuda.synchronize(device)

        elapses_across_iters.append(timer.get_elapses())
    stats = compute_elapses_stats(elapses_across_iters[num_warmup_iters:])
    log_elapses(stats, elapses_across_iters)


def get_batch() -> torch.Tensor:
    batch_size = TRANSFORMER_DEFAULT_BATCH_SIZE
    seq_len = TRANSFORMER_LARGE_SINGLE_PROCESS_CONTEXT_LENGTH
    vocab_size = TRANSFORMER_LARGE["vocab_size"]
    batch = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
    return batch


@pytest.mark.parametrize(
    "transformer_cls",
    [
        BlockwiseCheckpointedTransformer,
        AttnCheckpointedTransformer,
        FFNCheckpointedTransformer,
    ],
)
def test_transformer_actckpt(transformer_cls: Type[Transformer]):
    assert torch.cuda.device_count() >= 2, "This test requires at least 2 GPUs."
    set_seed()
    batch = get_batch()

    batch_size, sample_len = batch.shape
    seq_len = sample_len - 1
    cls_name = transformer_cls.__name__
    logger.info(f"=== Test {cls_name} ===")
    logger.info(
        f"Test Transformer activation checkpointing: args={TRANSFORMER_LARGE}, "
        f"batch_size={batch_size}, seq_len={seq_len}, transformer_cls={cls_name}"
    )

    state_dict = Transformer(
        device=torch.device("cpu"), **TRANSFORMER_LARGE
    ).state_dict()

    result = profile_transformer_ckpt_memory(
        state_dict,
        batch,
        torch.device("cuda:0"),
        Transformer,
        f"{cls_name}_nockpt.pickle",
    )
    result_ckpt = profile_transformer_ckpt_memory(
        state_dict,
        batch,
        torch.device("cuda:1"),
        transformer_cls,
        f"{cls_name}_ckpt.pickle",
    )

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
    logger.info(f"(Transformer) No checkpointing - pre-forward memory: {pre_fwd_mem}")
    logger.info(f"(Transformer) Checkpointing - pre-forward memory: {pre_fwd_mem_ckpt}")
    logger.info(f"(Transformer) No checkpointing - post-forward memory: {post_fwd_mem}")
    logger.info(
        f"(Transformer) Checkpointing - post-forward memory: {post_fwd_mem_ckpt}"
    )


if __name__ == "__main__":
    test_mlp_actckpt()
    for transformer_cls in [
        BlockwiseCheckpointedTransformer,
        AttnCheckpointedTransformer,
        FFNCheckpointedTransformer,
    ]:
        test_transformer_actckpt(transformer_cls)
    for transformer_cls in [
        Transformer,
        BlockwiseCheckpointedTransformer,
        AttnCheckpointedTransformer,
        FFNCheckpointedTransformer,
    ]:
        time_transformer_ckpt(
            get_batch(),
            torch.device("cuda:1"),
            transformer_cls,
        )
