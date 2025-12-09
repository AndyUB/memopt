import torch
import torch.nn as nn
import csv
import os
from datetime import datetime

from memopt.mixprec import MixedPrecisionTrainer
from memopt.model.transformer import Transformer
from memopt.model.optimizer import AdamW
from memopt.model.config import (
    TRANSFORMER_SMALL,
    TRANSFORMER_LARGE,
    TRANSFORMER_DEFAULT_BATCH_SIZE,
    DEFAULT_ADAMW_ARGS,
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
    """Benchmark latency comparing FP32 and three mixed precision configurations."""
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
    
    # ====================
    # FP32 (No Mixed Precision)
    # ====================
    print(f"\n{'='*60}")
    print("FP32 Training (No Mixed Precision)")
    print(f"{'='*60}")
    
    model_fp32 = Transformer(
        vocab_size=model_config["vocab_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=CONTEXT_LENGTH,
        device=device,
    )
    
    optimizer_fp32 = AdamW(model_fp32.parameters(), **DEFAULT_ADAMW_ARGS)
    
    # Generate data
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )
    
    # Warmup
    print(f"Warming up for {NUM_WARMUP_ITERS} iterations...")
    for step in range(NUM_WARMUP_ITERS):
        logits = model_fp32(token_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        loss.backward()
        optimizer_fp32.step()
        optimizer_fp32.zero_grad()
    
    # Benchmark using CUDA events
    print(f"Timing FP32 ({NUM_TIMING_ITERS} iterations)...")
    forward_times_fp32 = []
    backward_times_fp32 = []
    step_times_fp32 = []
    
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
        logits = model_fp32(token_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, model_config["vocab_size"]), targets.view(-1)
        )
        forward_end_events[step].record()
        
        # Backward pass
        backward_start_events[step].record()
        loss.backward()
        backward_end_events[step].record()
        
        # Optimizer step
        step_start_events[step].record()
        optimizer_fp32.step()
        optimizer_fp32.zero_grad()
        step_end_events[step].record()
        
        torch.cuda.synchronize()
    
    # Calculate timings after all iterations complete
    for step in range(NUM_TIMING_ITERS):
        forward_times_fp32.append(forward_start_events[step].elapsed_time(forward_end_events[step]) / 1000.0)
        backward_times_fp32.append(backward_start_events[step].elapsed_time(backward_end_events[step]) / 1000.0)
        step_times_fp32.append(step_start_events[step].elapsed_time(step_end_events[step]) / 1000.0)
    
    avg_forward_fp32 = sum(forward_times_fp32) / len(forward_times_fp32)
    avg_backward_fp32 = sum(backward_times_fp32) / len(backward_times_fp32)
    avg_step_fp32 = sum(step_times_fp32) / len(step_times_fp32)
    total_fp32 = avg_forward_fp32 + avg_backward_fp32 + avg_step_fp32
    
    print(f"FP32 breakdown (average per iteration):")
    print(f"  Forward:   {avg_forward_fp32:.4f} seconds ({avg_forward_fp32/total_fp32*100:.1f}%)")
    print(f"  Backward:  {avg_backward_fp32:.4f} seconds ({avg_backward_fp32/total_fp32*100:.1f}%)")
    print(f"  Optimizer: {avg_step_fp32:.4f} seconds ({avg_step_fp32/total_fp32*100:.1f}%)")
    print(f"  Total:     {total_fp32:.4f} seconds")
    
    # Clean up FP32
    del model_fp32, optimizer_fp32, logits, loss
    torch.cuda.empty_cache()
    
    # Helper function to benchmark a trainer configuration
    def benchmark_trainer(trainer, config_name, token_ids, targets, model_config):
        # Warmup
        print(f"Warming up for {NUM_WARMUP_ITERS} iterations...")
        for step in range(NUM_WARMUP_ITERS):
            (token_ids_cast,) = trainer.cast_inputs(token_ids)
            with trainer.forward_context():
                logits = trainer.model(token_ids_cast)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model_config["vocab_size"]), targets.view(-1)
            )
            scaled_loss = trainer.scale_loss(loss)
            scaled_loss.backward()
            trainer.step(check_overflow=False)
        
        # Benchmark using CUDA events
        print(f"Timing {config_name} ({NUM_TIMING_ITERS} iterations)...")
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
            with trainer.forward_context():
                logits = trainer.model(token_ids_cast)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, model_config["vocab_size"]), targets.view(-1)
            )
            forward_end_events[step].record()
            
            # Backward pass
            backward_start_events[step].record()
            scaled_loss = trainer.scale_loss(loss)
            scaled_loss.backward()
            backward_end_events[step].record()
            
            # Optimizer step
            step_start_events[step].record()
            trainer.step(check_overflow=False)
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
        total = avg_forward + avg_backward + avg_step
        
        print(f"{config_name} breakdown (average per iteration):")
        print(f"  Forward:   {avg_forward:.4f} seconds ({avg_forward/total*100:.1f}%)")
        print(f"  Backward:  {avg_backward:.4f} seconds ({avg_backward/total*100:.1f}%)")
        print(f"  Optimizer: {avg_step:.4f} seconds ({avg_step/total*100:.1f}%)")
        print(f"  Total:     {total:.4f} seconds")
        
        return avg_forward, avg_backward, avg_step, total
    
    # ====================
    # FP16 with autocast
    # ====================
    print(f"\n{'='*60}")
    print("FP16 Training (autocast)")
    print(f"{'='*60}")
    
    model_autocast = Transformer(
        vocab_size=model_config["vocab_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=CONTEXT_LENGTH,
        device=device,
    )
    
    trainer_autocast = MixedPrecisionTrainer(
        model_autocast,
        lower_precision_dtype=torch.float16,
        device=device,
        use_autocast=True,
        use_master_copy=True,
    )
    
    avg_forward_autocast, avg_backward_autocast, avg_step_autocast, total_autocast = benchmark_trainer(
        trainer_autocast, "FP16 (autocast)", token_ids, targets, model_config
    )
    
    del model_autocast, trainer_autocast
    torch.cuda.empty_cache()
    
    # ====================
    # FP16 without master copy
    # ====================
    print(f"\n{'='*60}")
    print("FP16 Training (no master)")
    print(f"{'='*60}")
    
    model_no_master = Transformer(
        vocab_size=model_config["vocab_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=CONTEXT_LENGTH,
        device=device,
    )
    
    trainer_no_master = MixedPrecisionTrainer(
        model_no_master,
        lower_precision_dtype=torch.float16,
        device=device,
        use_autocast=False,
        use_master_copy=False,
    )
    
    avg_forward_no_master, avg_backward_no_master, avg_step_no_master, total_no_master = benchmark_trainer(
        trainer_no_master, "FP16 (no master)", token_ids, targets, model_config
    )
    
    del model_no_master, trainer_no_master
    torch.cuda.empty_cache()
    
    # ====================
    # FP16 default
    # ====================
    print(f"\n{'='*60}")
    print("FP16 Training (default)")
    print(f"{'='*60}")
    
    model_fp16 = Transformer(
        vocab_size=model_config["vocab_size"],
        num_layers=model_config["num_layers"],
        d_model=model_config["d_model"],
        num_heads=model_config["num_heads"],
        d_ff=model_config["d_ff"],
        context_length=CONTEXT_LENGTH,
        device=device,
    )
    
    trainer_fp16 = MixedPrecisionTrainer(
        model_fp16,
        lower_precision_dtype=torch.float16,
        device=device,
    )
    
    avg_forward_fp16, avg_backward_fp16, avg_step_fp16, total_fp16 = benchmark_trainer(
        trainer_fp16, "FP16 (default)", token_ids, targets, model_config
    )
    
    del model_fp16, trainer_fp16
    torch.cuda.empty_cache()
    
    # Print comparison summary
    print(f"\n{'='*60}")
    print(f"Comparison Summary:")
    print(f"{'='*60}")
    print(f"Total time FP32:       {total_fp32:.4f} sec")
    print(f"Total time autocast:   {total_autocast:.4f} sec ({total_fp32/total_autocast:.2f}x)")
    print(f"Total time no_master:  {total_no_master:.4f} sec ({total_fp32/total_no_master:.2f}x)")
    print(f"Total time default:    {total_fp16:.4f} sec ({total_fp32/total_fp16:.2f}x)")
    
    # Store results
    result_row['total_fp32_sec'] = total_fp32
    result_row['total_autocast_sec'] = total_autocast
    result_row['total_no_master_sec'] = total_no_master
    result_row['total_default_sec'] = total_fp16
    result_row['speedup_autocast'] = total_fp32 / total_autocast
    result_row['speedup_no_master'] = total_fp32 / total_no_master
    result_row['speedup_default'] = total_fp32 / total_fp16
    
    results_list.append(result_row)


if __name__ == "__main__":
    print("Benchmarking Transformer models: FP32 vs FP16 Mixed Precision Configurations")
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
    csv_filename = os.path.join(output_dir, f"latency_mixprec_benchmark_{timestamp}.csv")
    
    if results:
        fieldnames = [
            'model_name', 'num_layers', 'd_model', 'num_heads', 'd_ff',
            'context_length', 'batch_size',
            'total_fp32_sec', 'total_autocast_sec', 
            'total_no_master_sec', 'total_default_sec',
            'speedup_autocast', 'speedup_no_master', 'speedup_default',
        ]
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*60}")
        print(f"Results exported to: {csv_filename}")
        print(f"{'='*60}")
        print(f"{'='*60}")
