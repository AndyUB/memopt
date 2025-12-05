import torch
import csv
import os
from datetime import datetime

from memopt.mixprec_offload import MixedPrecOffloadTrainer
from memopt.model.transformer import Transformer
from memopt.model.config import (
    TRANSFORMER_DEFAULT_BATCH_SIZE,
    OOM_TRANSFORMER_CONFIGS,
    CKPT_STRATEGIES,
)
from memopt.benchmark.mem_offload import (
    make_data,
    reset_memory_stats,
    get_gpu_memory_stats,
)


device = torch.device("cuda")
CONTEXT_LENGTH = 256
BATCH_SIZE = TRANSFORMER_DEFAULT_BATCH_SIZE
NUM_WARMUP_ITERS = 1
NUM_MEMORY_ITERS = 2


def benchmark_oom(model_config, model_name, results_list, transformer_cls=Transformer):
    """Measure peak GPU memory for a model with CPU offload."""
    print(f"\n{'='*60}")
    print(f"Memory Benchmarking {model_name}")
    print(f"{'='*60}")
    print(f"Config: {model_config}")
    print(f"Context length: {CONTEXT_LENGTH}, Batch size: {BATCH_SIZE}")
    print(f"transformer_cls: {transformer_cls.__name__}")

    result_row = {
        "model_name": model_name,
        "num_layers": model_config["num_layers"],
        "d_model": model_config["d_model"],
        "num_heads": model_config["num_heads"],
        "d_ff": model_config["d_ff"],
        "context_length": CONTEXT_LENGTH,
        "batch_size": BATCH_SIZE,
    }

    # =====================================================================
    # Benchmark WITH offload
    # =====================================================================
    print(f"\n--- WITH CPU Offload ---")
    reset_memory_stats()

    # Create model
    model = transformer_cls(
        vocab_size=model_config["vocab_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=CONTEXT_LENGTH,
        device=device,
    )

    after_model_offload = get_gpu_memory_stats()
    print(
        f"After model creation: {after_model_offload['allocated']:.2f} GB allocated, "
        f"{after_model_offload['reserved']:.2f} GB reserved"
    )

    # Create trainer with offload
    trainer_offload = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=torch.bfloat16,
        use_cpu_offload=True,
        use_grad_scaler=False,
    )

    after_trainer_offload = get_gpu_memory_stats()
    print(
        f"After trainer creation: {after_trainer_offload['allocated']:.2f} GB allocated, "
        f"{after_trainer_offload['reserved']:.2f} GB reserved"
    )

    # Generate data
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )

    after_data_offload = get_gpu_memory_stats()
    print(
        f"After data creation: {after_data_offload['allocated']:.2f} GB allocated, "
        f"{after_data_offload['reserved']:.2f} GB reserved"
    )

    # Warmup
    print(f"\nWarmup ({NUM_WARMUP_ITERS} iterations)...")
    for step in range(NUM_WARMUP_ITERS):
        (token_ids_cast,) = trainer_offload.cast_inputs(token_ids)
        logits = trainer_offload.forward(token_ids_cast)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        trainer_offload.backward(loss)
        ok = trainer_offload.step(check_overflow=True)

    after_warmup_offload = get_gpu_memory_stats()
    print(
        f"After warmup: {after_warmup_offload['allocated']:.2f} GB allocated, "
        f"{after_warmup_offload['reserved']:.2f} GB reserved"
    )
    print(
        f"Peak during warmup: {after_warmup_offload['max_allocated']:.2f} GB allocated, "
        f"{after_warmup_offload['max_reserved']:.2f} GB reserved"
    )

    # Reset peak stats before timing iterations
    reset_memory_stats()

    # Run timing iterations
    print(f"\nRunning {NUM_MEMORY_ITERS} iterations...")
    for step in range(NUM_MEMORY_ITERS):
        (token_ids_cast,) = trainer_offload.cast_inputs(token_ids)
        logits = trainer_offload.forward(token_ids_cast)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        trainer_offload.backward(loss)
        ok = trainer_offload.step(check_overflow=True)

    final_stats_with_offload = get_gpu_memory_stats()
    print(f"\nFinal memory usage:")
    print(
        f"  Current: {final_stats_with_offload['allocated']:.2f} GB allocated, "
        f"{final_stats_with_offload['reserved']:.2f} GB reserved"
    )
    print(
        f"  Peak:    {final_stats_with_offload['max_allocated']:.2f} GB allocated, "
        f"{final_stats_with_offload['max_reserved']:.2f} GB reserved"
    )

    # =====================================================================
    # Summary comparison
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"Memory Comparison Summary")
    print(f"{'='*60}")

    peak_with_offload = final_stats_with_offload["max_allocated"]

    print(f"\nPeak GPU Memory (allocated):")
    print(f"  With offload:    {peak_with_offload:.2f} GB")

    peak_reserved_with_offload = final_stats_with_offload["max_reserved"]

    print(f"\nPeak GPU Memory (reserved):")
    print(f"  With offload:    {peak_reserved_with_offload:.2f} GB")

    # Store results
    result_row["peak_allocated_with_offload_gb"] = peak_with_offload
    result_row["peak_reserved_with_offload_gb"] = peak_reserved_with_offload

    results_list.append(result_row)

    # Clean up
    del model, trainer_offload, token_ids, targets, logits, loss
    torch.cuda.empty_cache()

    return peak_with_offload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="all")
    parser.add_argument("--ckpt_strat", type=str, default="all")
    parser.add_argument("--result_csv", type=str, default=None)
    args = parser.parse_args()
    print(f"Arguments: {args}")

    print("GPU Memory Benchmarking: Transformer models with and without CPU offload")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(
        f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GiB"
    )

    # Store results for CSV export
    results = []

    if args.ckpt_strat == "all":
        ckpt_strats = ["Attention", "FFN", "Blockwise"]
    else:
        ckpt_strats = [args.ckpt_strat]

    if args.model_name == "all":
        model_names = list(OOM_TRANSFORMER_CONFIGS.keys())
    else:
        model_names = [args.model_name]

    result_rows = []
    for ckpt_strat in ckpt_strats:
        for model_name, model_config in OOM_TRANSFORMER_CONFIGS.items():
            # if model_name.startswith("large/") and model_name.endswith("layers"):
            #     num_layers = int(model_name[len("large/") : -len("layers")])
            #     if num_layers < 50:
            #         continue
            #     continue
            if model_name not in model_names:
                continue

            try:
                peak_with_offload = benchmark_oom(
                    model_config,
                    model_name,
                    results,
                    transformer_cls=CKPT_STRATEGIES[ckpt_strat],
                )
                result_rows.append(
                    {
                        "model_name": model_name,
                        "ckpt_strat": ckpt_strat,
                        "peak_allocated_with_offload_gib": peak_with_offload,
                    }
                )
            except RuntimeError as e:
                print(f"OOM? {e}")

        # Export results to CSV
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        csv_filename = os.path.join(
            output_dir, f"memory_benchmark_{ckpt_strat}_{timestamp}.csv"
        )

        if results:
            fieldnames = [
                "model_name",
                "num_layers",
                "d_model",
                "num_heads",
                "d_ff",
                "context_length",
                "batch_size",
                "peak_allocated_with_offload_gb",
                "peak_reserved_with_offload_gb",
            ]

            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results)

            print(f"\n{'='*60}")
            print(f"Results exported to: {csv_filename}")
            print(f"{'='*60}")

    if args.result_csv is not None:
        csv_filename = args.result_csv
        # If CSV file exists, append; otherwise, create new
        file_mode = "a" if os.path.exists(csv_filename) else "w"
        write_header = not os.path.exists(csv_filename)
        with open(csv_filename, file_mode, newline="") as csvfile:
            fieldnames = [
                "model_name",
                "ckpt_strat",
                "peak_allocated_with_offload_gib",
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerows(result_rows)
