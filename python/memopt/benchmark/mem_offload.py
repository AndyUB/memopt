import torch
import csv
import os
from datetime import datetime

from memopt.mixprec_offload import MixedPrecOffloadTrainer
from memopt.model.transformer import Transformer
from memopt.model.config import (
    TRANSFORMER_SMALL,
    TRANSFORMER_LARGE,
    TRANSFORMER_DEFAULT_BATCH_SIZE,
)


device = torch.device("cuda")
CONTEXT_LENGTH = 256
BATCH_SIZE = TRANSFORMER_DEFAULT_BATCH_SIZE
NUM_WARMUP_ITERS = 3
NUM_TIMING_ITERS = 5


def make_data(batch_size, context_length, vocab_size, device="cpu"):
    """Generate random token IDs and target logits for training."""
    token_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return token_ids, targets


def get_gpu_memory_stats():
    """Get current GPU memory usage statistics in GB."""
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    max_allocated = torch.cuda.max_memory_allocated() / 1024**3
    max_reserved = torch.cuda.max_memory_reserved() / 1024**3
    return {
        'allocated': allocated,
        'reserved': reserved,
        'max_allocated': max_allocated,
        'max_reserved': max_reserved,
    }


def reset_memory_stats():
    """Reset peak memory statistics."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()


def benchmark_memory(model_config, model_name, results_list):
    """Measure peak GPU memory for a model with and without CPU offload."""
    print(f"\n{'='*60}")
    print(f"Memory Benchmarking {model_name}")
    print(f"{'='*60}")
    print(f"Config: {model_config}")
    print(f"Context length: {CONTEXT_LENGTH}, Batch size: {BATCH_SIZE}")
    
    result_row = {
        'model_name': model_name,
        'num_layers': model_config['num_layers'],
        'd_model': model_config['d_model'],
        'num_heads': model_config['num_heads'],
        'd_ff': model_config['d_ff'],
        'context_length': CONTEXT_LENGTH,
        'batch_size': BATCH_SIZE,
    }
    
    # =====================================================================
    # Benchmark WITHOUT offload
    # =====================================================================
    print(f"\n--- WITHOUT CPU Offload ---")
    reset_memory_stats()
    
    # Create model
    model = Transformer(
        vocab_size=model_config["vocab_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=CONTEXT_LENGTH,
        device=device,
    )
    
    after_model = get_gpu_memory_stats()
    print(f"After model creation: {after_model['allocated']:.2f} GB allocated, "
          f"{after_model['reserved']:.2f} GB reserved")
    
    # Create trainer
    trainer = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=None,
        use_cpu_offload=False,
        use_grad_scaler=False,
    )
    
    after_trainer = get_gpu_memory_stats()
    print(f"After trainer creation: {after_trainer['allocated']:.2f} GB allocated, "
          f"{after_trainer['reserved']:.2f} GB reserved")
    
    # Generate data
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )
    
    after_data = get_gpu_memory_stats()
    print(f"After data creation: {after_data['allocated']:.2f} GB allocated, "
          f"{after_data['reserved']:.2f} GB reserved")
    
    # Warmup
    print(f"\nWarmup ({NUM_WARMUP_ITERS} iterations)...")
    for step in range(NUM_WARMUP_ITERS):
        (token_ids_cast,) = trainer.cast_inputs(token_ids)
        logits = trainer.forward(token_ids_cast)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        trainer.backward(loss)
        ok = trainer.step(check_overflow=True)
    
    after_warmup = get_gpu_memory_stats()
    print(f"After warmup: {after_warmup['allocated']:.2f} GB allocated, "
          f"{after_warmup['reserved']:.2f} GB reserved")
    print(f"Peak during warmup: {after_warmup['max_allocated']:.2f} GB allocated, "
          f"{after_warmup['max_reserved']:.2f} GB reserved")
    
    # Reset peak stats before timing iterations
    reset_memory_stats()
    
    # Run timing iterations
    print(f"\nRunning {NUM_TIMING_ITERS} iterations...")
    for step in range(NUM_TIMING_ITERS):
        (token_ids_cast,) = trainer.cast_inputs(token_ids)
        logits = trainer.forward(token_ids_cast)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        trainer.backward(loss)
        ok = trainer.step(check_overflow=True)
    
    final_stats_no_offload = get_gpu_memory_stats()
    print(f"\nFinal memory usage:")
    print(f"  Current: {final_stats_no_offload['allocated']:.2f} GB allocated, "
          f"{final_stats_no_offload['reserved']:.2f} GB reserved")
    print(f"  Peak:    {final_stats_no_offload['max_allocated']:.2f} GB allocated, "
          f"{final_stats_no_offload['max_reserved']:.2f} GB reserved")
    
    # Clean up
    del model, trainer, token_ids, targets, logits, loss
    torch.cuda.empty_cache()
    
    # =====================================================================
    # Benchmark WITH offload
    # =====================================================================
    print(f"\n--- WITH CPU Offload ---")
    reset_memory_stats()
    
    # Create model
    model = Transformer(
        vocab_size=model_config["vocab_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=CONTEXT_LENGTH,
        device=device,
    )
    
    after_model_offload = get_gpu_memory_stats()
    print(f"After model creation: {after_model_offload['allocated']:.2f} GB allocated, "
          f"{after_model_offload['reserved']:.2f} GB reserved")
    
    # Create trainer with offload
    trainer_offload = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=None,
        use_cpu_offload=True,
        use_grad_scaler=False,
    )
    
    after_trainer_offload = get_gpu_memory_stats()
    print(f"After trainer creation: {after_trainer_offload['allocated']:.2f} GB allocated, "
          f"{after_trainer_offload['reserved']:.2f} GB reserved")
    
    # Generate data
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )
    
    after_data_offload = get_gpu_memory_stats()
    print(f"After data creation: {after_data_offload['allocated']:.2f} GB allocated, "
          f"{after_data_offload['reserved']:.2f} GB reserved")
    
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
    print(f"After warmup: {after_warmup_offload['allocated']:.2f} GB allocated, "
          f"{after_warmup_offload['reserved']:.2f} GB reserved")
    print(f"Peak during warmup: {after_warmup_offload['max_allocated']:.2f} GB allocated, "
          f"{after_warmup_offload['max_reserved']:.2f} GB reserved")
    
    # Reset peak stats before timing iterations
    reset_memory_stats()
    
    # Run timing iterations
    print(f"\nRunning {NUM_TIMING_ITERS} iterations...")
    for step in range(NUM_TIMING_ITERS):
        (token_ids_cast,) = trainer_offload.cast_inputs(token_ids)
        logits = trainer_offload.forward(token_ids_cast)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        trainer_offload.backward(loss)
        ok = trainer_offload.step(check_overflow=True)
    
    final_stats_with_offload = get_gpu_memory_stats()
    print(f"\nFinal memory usage:")
    print(f"  Current: {final_stats_with_offload['allocated']:.2f} GB allocated, "
          f"{final_stats_with_offload['reserved']:.2f} GB reserved")
    print(f"  Peak:    {final_stats_with_offload['max_allocated']:.2f} GB allocated, "
          f"{final_stats_with_offload['max_reserved']:.2f} GB reserved")
    
    # =====================================================================
    # Summary comparison
    # =====================================================================
    print(f"\n{'='*60}")
    print(f"Memory Comparison Summary")
    print(f"{'='*60}")
    
    peak_no_offload = final_stats_no_offload['max_allocated']
    peak_with_offload = final_stats_with_offload['max_allocated']
    memory_saved = peak_no_offload - peak_with_offload
    memory_reduction_pct = (memory_saved / peak_no_offload) * 100 if peak_no_offload > 0 else 0
    
    print(f"\nPeak GPU Memory (allocated):")
    print(f"  Without offload: {peak_no_offload:.2f} GB")
    print(f"  With offload:    {peak_with_offload:.2f} GB")
    print(f"  Memory saved:    {memory_saved:.2f} GB ({memory_reduction_pct:.1f}% reduction)")
    
    peak_reserved_no_offload = final_stats_no_offload['max_reserved']
    peak_reserved_with_offload = final_stats_with_offload['max_reserved']
    reserved_saved = peak_reserved_no_offload - peak_reserved_with_offload
    reserved_reduction_pct = (reserved_saved / peak_reserved_no_offload) * 100 if peak_reserved_no_offload > 0 else 0
    
    print(f"\nPeak GPU Memory (reserved):")
    print(f"  Without offload: {peak_reserved_no_offload:.2f} GB")
    print(f"  With offload:    {peak_reserved_with_offload:.2f} GB")
    print(f"  Memory saved:    {reserved_saved:.2f} GB ({reserved_reduction_pct:.1f}% reduction)")
    
    # Store results
    result_row['peak_allocated_no_offload_gb'] = peak_no_offload
    result_row['peak_allocated_with_offload_gb'] = peak_with_offload
    result_row['memory_saved_allocated_gb'] = memory_saved
    result_row['memory_reduction_allocated_pct'] = memory_reduction_pct
    result_row['peak_reserved_no_offload_gb'] = peak_reserved_no_offload
    result_row['peak_reserved_with_offload_gb'] = peak_reserved_with_offload
    result_row['memory_saved_reserved_gb'] = reserved_saved
    result_row['memory_reduction_reserved_pct'] = reserved_reduction_pct
    
    results_list.append(result_row)
    
    # Clean up
    del model, trainer_offload, token_ids, targets, logits, loss
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("GPU Memory Benchmarking: Transformer models with and without CPU offload")
    print(f"Device: {device}")
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.2f} GB")
    
    # Store results for CSV export
    results = []
    
    # Benchmark TRANSFORMER_SMALL
    benchmark_memory(TRANSFORMER_SMALL, "TRANSFORMER_SMALL", results)
    
    # Benchmark TRANSFORMER_LARGE
    benchmark_memory(TRANSFORMER_LARGE, "TRANSFORMER_LARGE", results)
    
    # Export results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"memory_benchmark_{timestamp}.csv")
    
    if results:
        fieldnames = [
            'model_name', 'num_layers', 'd_model', 'num_heads', 'd_ff',
            'context_length', 'batch_size',
            'peak_allocated_no_offload_gb', 'peak_allocated_with_offload_gb',
            'memory_saved_allocated_gb', 'memory_reduction_allocated_pct',
            'peak_reserved_no_offload_gb', 'peak_reserved_with_offload_gb',
            'memory_saved_reserved_gb', 'memory_reduction_reserved_pct',
        ]
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*60}")
        print(f"Results exported to: {csv_filename}")
        print(f"{'='*60}")
