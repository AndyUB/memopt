from argparse import ArgumentParser
import csv
import os
import torch


from memopt.model.transformer import Transformer
from memopt.model.optimizer import AdamW, cross_entropy_loss
from memopt.model.config import (
    TRAINABLE_TRANSFORMER_CONFIGS,
    OOM_TRANSFORMER_CONFIGS,
    TRANSFORMER_DEFAULT_BATCH_SIZE as BATCH_SIZE,
    TRANSFORMER_LARGE_SINGLE_PROCESS_CONTEXT_LENGTH as CONTEXT_LENGTH,
    DEFAULT_ADAMW_ARGS,
    CKPT_STRATEGIES,
)
from memopt.util import get_timing_events, set_seed, measure_peak_memory
from memopt.offload import CPUOffloadTrainer
from memopt.zero_offload import ZeroOffloadTrainer
from memopt.mixprec import MixedPrecisionTrainer

TRANSFORMER_CONFIGS = {**TRAINABLE_TRANSFORMER_CONFIGS, **OOM_TRANSFORMER_CONFIGS}
NONE_CKPT_STRATEGY = "None"


class BenchmarkRunner:
    def __init__(
        self,
        benchmark_name: str,
        model_config_name: str,
        model_cls: type[Transformer],
        device: torch.device | None = None,
    ):
        self.benchmark_name = benchmark_name
        self.model_config_name = model_config_name
        self.model_cls = model_cls
        self.device = device if device is not None else torch.device("cuda")

        self.model_cls_name = model_cls.__name__
        self.model_config = TRANSFORMER_CONFIGS[model_config_name]
        self.batch_size = BATCH_SIZE
        self.context_length = CONTEXT_LENGTH

        self.model = model_cls(
            device=self.device,
            **self.model_config,
        ).to(self.device)

    def setup(self):
        self.optimizer = AdamW(self.model.parameters(), **DEFAULT_ADAMW_ARGS)

    def train_iter(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        output = self.model(x)
        loss = cross_entropy_loss(output, y)
        loss.backward()
        self.optimizer.step()

    def run(self, num_warmup_iters: int = 3, num_benchmark_iters: int = 10) -> None:
        self.setup()

        set_seed()
        batch = torch.randint(
            0,
            self.model_config["vocab_size"],
            (self.batch_size, self.context_length + 1),
            device=self.device,
        )
        input_ids = batch[:, :-1]
        target_ids = batch[:, 1:]

        # Warm-up
        for _ in range(num_warmup_iters):
            self.train_iter(input_ids, target_ids)

        # Benchmark
        start_ev, end_ev = get_timing_events(num_events=2)
        measure_peak_memory(self.device)

        torch.cuda.synchronize(self.device)
        start_ev.record()
        for _ in range(num_benchmark_iters):
            self.train_iter(input_ids, target_ids)
        end_ev.record()
        torch.cuda.synchronize(self.device)

        mem_stats = measure_peak_memory(self.device)
        peak_mem_bytes = mem_stats.max_allocated_bytes
        elapse_ms = start_ev.elapsed_time(end_ev)
        avg_elapse_ms = elapse_ms / num_benchmark_iters

        result_file = f"pareto_{self.model_config_name.replace('/', '_')}.csv"
        if not os.path.exists(result_file):
            with open(result_file, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    ["benchmark_name", "model_cls", "avg_elapse_ms", "peak_mem_bytes"]
                )
        with open(result_file, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.benchmark_name,
                    self.model_cls_name,
                    avg_elapse_ms,
                    peak_mem_bytes,
                ]
            )


class BaselineRunner(BenchmarkRunner):
    def __init__(self, model_config_name: str) -> None:
        super().__init__(
            benchmark_name="Baseline",
            model_config_name=model_config_name,
            model_cls=Transformer,
        )


class ActckptRunner(BenchmarkRunner):
    def __init__(self, model_config_name: str, ckpt_strategy: str) -> None:
        if ckpt_strategy == NONE_CKPT_STRATEGY:
            raise ValueError("ckpt_strategy must not be 'None' for ActckptRunner")
        model_cls = CKPT_STRATEGIES[ckpt_strategy]
        benchmark_name = f"Activation Checkpointing ({ckpt_strategy})"
        super().__init__(
            benchmark_name=benchmark_name,
            model_config_name=model_config_name,
            model_cls=model_cls,
        )


class ZeroOffloadRunner(BenchmarkRunner):
    def __init__(
        self, model_config_name: str, ckpt_strategy: str = NONE_CKPT_STRATEGY
    ) -> None:
        benchmark_name = "CPU Offloading + Mixed Precision (FP16)"
        if ckpt_strategy != NONE_CKPT_STRATEGY:
            benchmark_name += f" + Activation Checkpointing ({ckpt_strategy})"
        model_cls = CKPT_STRATEGIES[ckpt_strategy]
        super().__init__(
            benchmark_name=benchmark_name,
            model_config_name=model_config_name,
            model_cls=model_cls,
        )

    def setup(self):
        self.trainer = ZeroOffloadTrainer(
            model=self.model,
            dtype=torch.float16,
            device=self.device,
            loss_scale=None,
        )

    def train_iter(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.trainer.train_iter(x, y, cross_entropy_loss, check_overflow=False)


class OffloadRunner(BenchmarkRunner):
    def __init__(
        self, model_config_name: str, ckpt_strategy: str = NONE_CKPT_STRATEGY
    ) -> None:
        benchmark_name = "CPU Offloading"
        if ckpt_strategy != NONE_CKPT_STRATEGY:
            benchmark_name += f" + Activation Checkpointing ({ckpt_strategy})"
        model_cls = CKPT_STRATEGIES[ckpt_strategy]
        super().__init__(
            benchmark_name=benchmark_name,
            model_config_name=model_config_name,
            model_cls=model_cls,
        )

    def setup(self):
        self.trainer = CPUOffloadTrainer(model=self.model, device=self.device)

    def train_iter(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.trainer.train_step(x, y, cross_entropy_loss)


class MixprecRunner(BenchmarkRunner):
    def __init__(
        self, model_config_name: str, ckpt_strategy: str = NONE_CKPT_STRATEGY
    ) -> None:
        benchmark_name = "Mixed Precision (FP16)"
        if ckpt_strategy != NONE_CKPT_STRATEGY:
            benchmark_name += f" + Activation Checkpointing ({ckpt_strategy})"
        model_cls = CKPT_STRATEGIES[ckpt_strategy]
        super().__init__(
            benchmark_name=benchmark_name,
            model_config_name=model_config_name,
            model_cls=model_cls,
        )

    def setup(self):
        self.trainer = MixedPrecisionTrainer(
            model=self.model,
            lower_precision_dtype=torch.float16,
            device=self.device,
            use_master_copy=True,
            loss_scale=None,
            use_autocast=True,
        )

    def train_iter(self, x: torch.Tensor, y: torch.Tensor) -> None:
        x, y = self.trainer.cast_inputs(x, y)
        with self.trainer.forward_context():
            output = self.model(x)
            loss = cross_entropy_loss(output, y)
        scaled_loss = self.trainer.scale_loss(loss)
        scaled_loss.backward()
        self.trainer.step(check_overflow=False)


if __name__ == "__main__":
    runner_classes: list[type[BenchmarkRunner]] = [
        BaselineRunner,
        ActckptRunner,
        MixprecRunner,
        OffloadRunner,
        ZeroOffloadRunner,
    ]
    named_runner_classes = {cls.__name__: cls for cls in runner_classes}
    parser = ArgumentParser()
    parser.add_argument(
        "--runner_cls",
        type=str,
        choices=list(named_runner_classes.keys()),
        required=True,
        help="Runner class for benchmarking",
    )
    parser.add_argument(
        "--model_config_name",
        type=str,
        choices=list(TRANSFORMER_CONFIGS.keys()),
        required=True,
        help="Model configuration name",
    )
    parser.add_argument(
        "--ckpt_strategy",
        type=str,
        choices=list(CKPT_STRATEGIES.keys()),
        default=None,
        help="Activation checkpointing strategy",
    )
    args = parser.parse_args()

    optional_args = {}
    if args.ckpt_strategy is not None:
        optional_args["ckpt_strategy"] = args.ckpt_strategy

    runner_cls = named_runner_classes[args.runner_cls]
    runner = runner_cls(
        model_config_name=args.model_config_name,
        **optional_args,
    )
    runner.run()
