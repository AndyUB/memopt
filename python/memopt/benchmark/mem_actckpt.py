import logging
import torch
from typing import Type

from memopt.model.config import DEFAULT_ADAMW_ARGS
from memopt.model.optimizer import AdamW, cross_entropy_loss
from memopt.model.transformer import Transformer
from memopt.util import MemoryStats, measure_peak_memory, set_seed
from memopt.benchmark.latency_actckpt import (
    CKPT_STRATEGIES,
    ModelSize,
    BATCH_SIZE,
    CONTEXT_LENGTH,
    DEVICE,
    CONFIGS,
)
from memopt.benchmark.benchmark_util import plot_ckpt_memory, to_df

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)


def benchmark_transformer_ckpt_memory(
    transformer_cls: Type[Transformer], model_size: ModelSize
) -> MemoryStats:
    device = DEVICE
    batch_size = BATCH_SIZE
    seq_len = CONTEXT_LENGTH
    model_config = CONFIGS[model_size]
    cls_name = transformer_cls.__name__

    logger.info(f"=== Benchmark {cls_name} memory ===")
    logger.info(
        f"Profile memory for Transformer activation checkpointing: "
        f"model_size={model_size}, model_config={model_config}, "
        f"batch_size={batch_size}, seq_len={seq_len}, transformer_cls={cls_name}"
    )

    set_seed()
    model = transformer_cls(
        device=device,
        **model_config,
    ).to(device)
    optimizer = AdamW(model.parameters(), **DEFAULT_ADAMW_ARGS)

    batch = torch.randint(
        low=0, high=model_config["vocab_size"], size=(batch_size, seq_len + 1)
    ).to(device)
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]

    mem_base = measure_peak_memory(device)
    for _ in range(2):
        optimizer.zero_grad()
        output = model(input_ids)
        loss = cross_entropy_loss(output, target_ids)
        loss.backward()
        optimizer.step()
    mem = measure_peak_memory(device)
    logger.info(f"Memory usage: base={mem_base}, peak={mem}")
    return mem


def compare_ckpt_memory(model_size: ModelSize) -> dict[str, float]:
    strat_to_stats: dict[str, float] = {}
    for strat, cls in CKPT_STRATEGIES.items():
        stats = benchmark_transformer_ckpt_memory(
            transformer_cls=cls,
            model_size=model_size,
        )
        strat_to_stats[strat] = stats.max_allocated_bytes
    return strat_to_stats


if __name__ == "__main__":
    config_to_stats: dict[str, dict[str, float]] = {}
    for model_size in CONFIGS.keys():
        strat_to_stats = compare_ckpt_memory(model_size=model_size)
        config_to_stats[model_size] = strat_to_stats

    plot_ckpt_memory(config_to_stats, "memory_actckpt.png")
    flat_stats = [{"model_size": k, **v} for k, v in config_to_stats.items()]
    to_df(flat_stats, "memory_actckpt.csv")
