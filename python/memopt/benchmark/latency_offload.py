import torch
import torch.nn as nn
import time
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


def benchmark_model(model_config, model_name, results_list):
    """Benchmark a model with and without CPU offload."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {model_name}")
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
    
    # Create trainers
    trainer = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=None,
        use_cpu_offload=False,
        use_grad_scaler=False,
    )
    
    trainer_offload = MixedPrecOffloadTrainer(
        model,
        lower_precision_dtype=None,
        use_cpu_offload=True,
        use_grad_scaler=False,
    )
    
    # Generate data
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )
    
    # Warmup (without offload)
    print(f"Warming up for {NUM_WARMUP_ITERS} iterations...")
    for step in range(NUM_WARMUP_ITERS):
        (token_ids_cast,) = trainer.cast_inputs(token_ids)
        logits = trainer.forward(token_ids_cast)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        trainer.backward(loss)
        ok = trainer.step(check_overflow=True)
    
    # Benchmark without offload using CUDA events
    print(f"\nTiming without offload ({NUM_TIMING_ITERS} iterations)...")
    forward_times = []
    backward_times = []
    step_times = []
    
    # Create all CUDA events upfront
    forward_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    forward_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    backward_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    backward_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    step_start_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    step_end_events = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    
    for step in range(NUM_TIMING_ITERS):
        torch.cuda.synchronize()
        
        # Forward pass
        forward_start_events[step].record()
        (token_ids_cast,) = trainer.cast_inputs(token_ids)
        logits = trainer.forward(token_ids_cast)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        forward_end_events[step].record()
        
        # Backward pass
        backward_start_events[step].record()
        trainer.backward(loss)
        backward_end_events[step].record()
        
        # Optimizer step
        step_start_events[step].record()
        ok = trainer.step(check_overflow=True)
        step_end_events[step].record()
        
        torch.cuda.synchronize()
    
    # Calculate timings after all iterations complete
    for step in range(NUM_TIMING_ITERS):
        forward_times.append(forward_start_events[step].elapsed_time(forward_end_events[step]) / 1000.0)
        backward_times.append(backward_start_events[step].elapsed_time(backward_end_events[step]) / 1000.0)
        step_times.append(step_start_events[step].elapsed_time(step_end_events[step]) / 1000.0)
    
    avg_forward = sum(forward_times) / len(forward_times)
    avg_backward = sum(backward_times) / len(backward_times)
    avg_step = sum(step_times) / len(step_times)
    total_no_offload = avg_forward + avg_backward + avg_step
    
    print(f"Without offload breakdown (average per iteration):")
    print(f"  Forward:   {avg_forward:.4f} seconds ({avg_forward/total_no_offload*100:.1f}%)")
    print(f"  Backward:  {avg_backward:.4f} seconds ({avg_backward/total_no_offload*100:.1f}%)")
    print(f"  Optimizer: {avg_step:.4f} seconds ({avg_step/total_no_offload*100:.1f}%)")
    print(f"  Total:     {total_no_offload:.4f} seconds")
    
    # Benchmark with offload using CUDA events
    print(f"\nTiming with CPU offload ({NUM_TIMING_ITERS} iterations)...")
    forward_times_offload = []
    backward_times_offload = []
    step_times_offload = []
    
    # Create all CUDA events upfront
    forward_start_events_offload = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    forward_end_events_offload = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    backward_start_events_offload = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    backward_end_events_offload = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    step_start_events_offload = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    step_end_events_offload = [torch.cuda.Event(enable_timing=True) for _ in range(NUM_TIMING_ITERS)]
    
    for step in range(NUM_TIMING_ITERS):
        torch.cuda.synchronize()
        
        # Forward pass
        forward_start_events_offload[step].record()
        (token_ids_cast,) = trainer_offload.cast_inputs(token_ids)
        logits = trainer_offload.forward(token_ids_cast)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        forward_end_events_offload[step].record()
        
        # Backward pass
        backward_start_events_offload[step].record()
        trainer_offload.backward(loss)
        backward_end_events_offload[step].record()
        
        # Optimizer step
        step_start_events_offload[step].record()
        ok = trainer_offload.step(check_overflow=True)
        step_end_events_offload[step].record()
        
        torch.cuda.synchronize()
    
    # Calculate timings after all iterations complete
    for step in range(NUM_TIMING_ITERS):
        forward_times_offload.append(forward_start_events_offload[step].elapsed_time(forward_end_events_offload[step]) / 1000.0)
        backward_times_offload.append(backward_start_events_offload[step].elapsed_time(backward_end_events_offload[step]) / 1000.0)
        step_times_offload.append(step_start_events_offload[step].elapsed_time(step_end_events_offload[step]) / 1000.0)
    
    avg_forward_offload = sum(forward_times_offload) / len(forward_times_offload)
    avg_backward_offload = sum(backward_times_offload) / len(backward_times_offload)
    avg_step_offload = sum(step_times_offload) / len(step_times_offload)
    total_with_offload = avg_forward_offload + avg_backward_offload + avg_step_offload
    
    print(f"With CPU offload breakdown (average per iteration):")
    print(f"  Forward:   {avg_forward_offload:.4f} seconds ({avg_forward_offload/total_with_offload*100:.1f}%)")
    print(f"  Backward:  {avg_backward_offload:.4f} seconds ({avg_backward_offload/total_with_offload*100:.1f}%)")
    print(f"  Optimizer: {avg_step_offload:.4f} seconds ({avg_step_offload/total_with_offload*100:.1f}%)")
    print(f"  Total:     {total_with_offload:.4f} seconds")
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print(f"Comparison Summary:")
    print(f"{'='*60}")
    speedup = total_no_offload / total_with_offload
    print(f"Overall speedup with offload: {speedup:.2f}x")
    if speedup < 1:
        slowdown = total_with_offload / total_no_offload
        print(f"(Overall slowdown: {slowdown:.2f}x)")
    
    print(f"\nPer-component speedup:")
    print(f"  Forward:   {avg_forward / avg_forward_offload:.2f}x")
    print(f"  Backward:  {avg_backward / avg_backward_offload:.2f}x")
    print(f"  Optimizer: {avg_step / avg_step_offload:.2f}x")
    
    # Store results
    result_row['forward_no_offload_sec'] = avg_forward
    result_row['backward_no_offload_sec'] = avg_backward
    result_row['optimizer_no_offload_sec'] = avg_step
    result_row['total_no_offload_sec'] = total_no_offload
    result_row['forward_with_offload_sec'] = avg_forward_offload
    result_row['backward_with_offload_sec'] = avg_backward_offload
    result_row['optimizer_with_offload_sec'] = avg_step_offload
    result_row['total_with_offload_sec'] = total_with_offload
    result_row['forward_speedup'] = avg_forward / avg_forward_offload
    result_row['backward_speedup'] = avg_backward / avg_backward_offload
    result_row['optimizer_speedup'] = avg_step / avg_step_offload
    result_row['overall_speedup'] = speedup
    
    results_list.append(result_row)


if __name__ == "__main__":
    print("Benchmarking Transformer models with and without CPU offload")
    print(f"Device: {device}")
    
    # Store results for CSV export
    results = []
    
    # Benchmark TRANSFORMER_SMALL
    benchmark_model(TRANSFORMER_SMALL, "TRANSFORMER_SMALL", results)
    
    # Benchmark TRANSFORMER_LARGE
    benchmark_model(TRANSFORMER_LARGE, "TRANSFORMER_LARGE", results)
    
    # Export results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = "results"
    os.makedirs(output_dir, exist_ok=True)
    csv_filename = os.path.join(output_dir, f"latency_benchmark_{timestamp}.csv")
    
    if results:
        fieldnames = [
            'model_name', 'num_layers', 'd_model', 'num_heads', 'd_ff',
            'context_length', 'batch_size',
            'forward_no_offload_sec', 'backward_no_offload_sec', 
            'optimizer_no_offload_sec', 'total_no_offload_sec',
            'forward_with_offload_sec', 'backward_with_offload_sec',
            'optimizer_with_offload_sec', 'total_with_offload_sec',
            'forward_speedup', 'backward_speedup', 'optimizer_speedup',
            'overall_speedup',
        ]
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*60}")
        print(f"Results exported to: {csv_filename}")
        print(f"{'='*60}")