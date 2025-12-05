import gc
import logging
import torch
from typing import Type, Any

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


def cuda_free():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


def benchmark_transformer_ckpt_memory(
    transformer_cls: Type[Transformer],
    model_size: ModelSize,
    model_config: dict[str, Any],
) -> MemoryStats:
    device = DEVICE
    batch_size = BATCH_SIZE
    seq_len = CONTEXT_LENGTH
    cls_name = transformer_cls.__name__

    logger.info(f"=== Benchmark {cls_name} memory ===")
    logger.info(
        f"Profile memory for Transformer activation checkpointing: "
        f"model_size={model_size}, model_config={model_config}, "
        f"batch_size={batch_size}, seq_len={seq_len}, transformer_cls={cls_name}"
    )

    # cuda_free()
    set_seed()
    mem_init = measure_peak_memory(device)
    logger.info(f"Initial memory usage: {mem_init}")
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


def compare_ckpt_memory(
    strats: dict[str, Type[Transformer]],
    configs: dict[str, Any],
) -> dict[str, dict[str, float]]:
    config_to_stats: dict[str, dict[str, float]] = {}
    for model_size in configs.keys():
        strat_to_stats: dict[str, float] = {}
        for strat, cls in strats.items():
            try:
                stats = benchmark_transformer_ckpt_memory(
                    transformer_cls=cls,
                    model_size=model_size,
                    model_config=configs[model_size],
                )
                strat_to_stats[strat] = stats.max_allocated_bytes
            except RuntimeError as e:
                logger.info(f"OOM? {e}")
        config_to_stats[model_size] = strat_to_stats
    return config_to_stats


if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "profile"

    if mode == "profile":
        configs = CONFIGS
        strats = CKPT_STRATEGIES
        result_name = "memory_actckpt"
    else:
        from memopt.model.config import OOM_TRANSFORMER_CONFIGS

        configs = OOM_TRANSFORMER_CONFIGS
        strats = CKPT_STRATEGIES.copy()
        # strats.pop("None")
        result_name = "memory_actckpt_oom"

    config_to_stats = compare_ckpt_memory(strats=strats, configs=configs)
    plot_ckpt_memory(config_to_stats, f"{result_name}.png")
    flat_stats = [{"model_size": k, **v} for k, v in config_to_stats.items()]
    to_df(flat_stats, f"{result_name}.csv")
