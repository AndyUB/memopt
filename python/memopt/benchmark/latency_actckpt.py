import logging
import pandas as pd
import torch
from typing import Any, Literal, Type

from memopt.benchmark.benchmark_util import plot_ckpt_times, to_df
from memopt.model.config import (
    DEFAULT_ADAMW_ARGS,
    TRANSFORMER_DEFAULT_BATCH_SIZE,
    TRANSFORMER_SMALL,
    TRANSFORMER_LARGE,
    TRANSFORMER_LARGE_SINGLE_PROCESS_CONTEXT_LENGTH,
)
from memopt.model.optimizer import AdamW, cross_entropy_loss
from memopt.model.ckpt_transformer import (
    AttnCheckpointedTransformer,
    BlockwiseCheckpointedTransformer,
    FFNCheckpointedTransformer,
)
from memopt.model.transformer import Transformer
from memopt.util import (
    compute_elapses_stats,
    log_elapses,
    set_seed,
    TrainingTimer,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

ModelSize = Literal["small", "large"]

CONTEXT_LENGTH = TRANSFORMER_LARGE_SINGLE_PROCESS_CONTEXT_LENGTH
BATCH_SIZE = TRANSFORMER_DEFAULT_BATCH_SIZE
NUM_WARMUP_ITERS = 5
NUM_TIMING_ITERS = 10
CONFIGS = {
    "small": TRANSFORMER_SMALL,
    "large": TRANSFORMER_LARGE,
}
DEVICE = torch.device("cuda")
CKPT_STRATEGIES: dict[str, Type[Transformer]] = {
    "Attention": AttnCheckpointedTransformer,
    "FFN": FFNCheckpointedTransformer,
    "Blockwise": BlockwiseCheckpointedTransformer,
    "None": Transformer,
}


def benchmark_transformer_ckpt_latency(
    transformer_cls: Type[Transformer],
    model_size: ModelSize,
) -> dict[str, dict[str, float]]:
    cls_name = transformer_cls.__name__
    model_config = CONFIGS[model_size]
    batch_size = BATCH_SIZE
    seq_len = CONTEXT_LENGTH
    device = DEVICE
    num_warmup_iters = NUM_WARMUP_ITERS
    num_benchmark_iters = NUM_TIMING_ITERS

    logger.info(f"=== Benchmark {cls_name} latency ===")
    logger.info(
        f"Time Transformer activation checkpointing: model_size={model_size}, "
        f"model_config={model_config}, batch_size={batch_size}, seq_len={seq_len}, "
        f"transformer_cls={cls_name}, num_warmup_iters={num_warmup_iters}, "
        f"num_benchmark_iters={num_benchmark_iters}"
    )

    set_seed()
    model = transformer_cls(
        device=device,
        **model_config,
    ).to(device)
    optimizer = AdamW(model.parameters(), **DEFAULT_ADAMW_ARGS)

    batch = torch.randint(
        low=0,
        high=model_config["vocab_size"],
        size=(batch_size, seq_len + 1),
    ).to(device)
    input_ids = batch[:, :-1]
    target_ids = batch[:, 1:]

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
    return stats


def compare_ckpt_latency(model_size: ModelSize) -> pd.DataFrame:
    strat_to_stats: dict[str, dict[str, float]] = {}
    for strat, cls in CKPT_STRATEGIES.items():
        stats = benchmark_transformer_ckpt_latency(
            transformer_cls=cls,
            model_size=model_size,
        )
        stats = {k: v["avg"] for k, v in stats.items()}
        stats["other"] = (
            stats["total"] - stats["forward"] - stats["backward"] - stats["step"]
        )
        strat_to_stats[strat] = stats

    data = [{"ckpt_strat": strat, **stats} for strat, stats in strat_to_stats.items()]
    df = to_df(data, f"latency_ckpt_{model_size}.csv")
    return df


if __name__ == "__main__":
    dfs: dict[str, pd.DataFrame] = {}
    for model_size in CONFIGS.keys():
        dfs[model_size] = compare_ckpt_latency(model_size=model_size)
    plot_ckpt_times(dfs, "latency_actckpt.png")
