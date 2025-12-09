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


def make_data(batch_size, context_length, vocab_size, device="cpu"):
    """Generate random token IDs and target logits for training."""
    token_ids = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, context_length), device=device)
    return token_ids, targets


def benchmark_model(model_config, model_name, results_list):
    """Benchmark memory usage comparing FP32 and three mixed precision configurations."""
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
    
    # Test FP32 (no mixed precision - normal training)
    print("\nTesting FP32 (no mixed precision)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
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
    
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )
    
    # Measure memory after model creation
    mem_after_model_fp32 = torch.cuda.max_memory_allocated() / (10**9)
    
    # Run one forward-backward pass
    logits = model_fp32(token_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, model_config["vocab_size"]), targets.view(-1)
    )
    
    # Measure memory after forward
    mem_after_forward_fp32 = torch.cuda.max_memory_allocated() / (10**9)
    
    loss.backward()
    
    # Measure memory after backward
    mem_after_backward_fp32 = torch.cuda.max_memory_allocated() / (10**9)
    
    optimizer_fp32.step()
    optimizer_fp32.zero_grad()
    
    # Peak memory
    peak_mem_fp32 = torch.cuda.max_memory_allocated() / (10**9)
    
    print(f"FP32 Memory Usage:")
    print(f"  After model:    {mem_after_model_fp32:.3f} GB")
    print(f"  After forward:  {mem_after_forward_fp32:.3f} GB")
    print(f"  After backward: {mem_after_backward_fp32:.3f} GB")
    print(f"  Peak:           {peak_mem_fp32:.3f} GB")
    
    # Clean up
    del model_fp32, optimizer_fp32, logits, loss, token_ids, targets
    torch.cuda.empty_cache()
    
    # Test FP16 with autocast only (use_autocast=True, use_master_copy=True)
    print("\nTesting FP16 (autocast only)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
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
    
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )
    
    # Measure memory after model creation
    mem_after_model_autocast = torch.cuda.max_memory_allocated() / (10**9)
    
    # Run one forward-backward pass
    (token_ids_cast,) = trainer_autocast.cast_inputs(token_ids)
    with trainer_autocast.forward_context():
        logits = trainer_autocast.model(token_ids_cast)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, model_config["vocab_size"]), targets.view(-1)
    )
    
    # Measure memory after forward
    mem_after_forward_autocast = torch.cuda.max_memory_allocated() / (10**9)
    
    scaled_loss = trainer_autocast.scale_loss(loss)
    scaled_loss.backward()
    
    # Measure memory after backward
    mem_after_backward_autocast = torch.cuda.max_memory_allocated() / (10**9)
    
    trainer_autocast.step(check_overflow=False)
    
    # Peak memory
    peak_mem_autocast = torch.cuda.max_memory_allocated() / (10**9)
    
    print(f"FP16 (autocast only) Memory Usage:")
    print(f"  After model:    {mem_after_model_autocast:.3f} GB")
    print(f"  After forward:  {mem_after_forward_autocast:.3f} GB")
    print(f"  After backward: {mem_after_backward_autocast:.3f} GB")
    print(f"  Peak:           {peak_mem_autocast:.3f} GB")
    
    # Clean up
    del model_autocast, trainer_autocast, logits, loss, token_ids, targets, token_ids_cast
    torch.cuda.empty_cache()
    
    # Test FP16 without master copy (use_autocast=False, use_master_copy=False)
    print("\nTesting FP16 (no master copy)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
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
    
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )
    
    # Measure memory after model creation
    mem_after_model_no_master = torch.cuda.max_memory_allocated() / (10**9)
    
    # Run one forward-backward pass
    (token_ids_cast,) = trainer_no_master.cast_inputs(token_ids)
    with trainer_no_master.forward_context():
        logits = trainer_no_master.model(token_ids_cast)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, model_config["vocab_size"]), targets.view(-1)
    )
    
    # Measure memory after forward
    mem_after_forward_no_master = torch.cuda.max_memory_allocated() / (10**9)
    
    scaled_loss = trainer_no_master.scale_loss(loss)
    scaled_loss.backward()
    
    # Measure memory after backward
    mem_after_backward_no_master = torch.cuda.max_memory_allocated() / (10**9)
    
    trainer_no_master.step(check_overflow=False)
    
    # Peak memory
    peak_mem_no_master = torch.cuda.max_memory_allocated() / (10**9)
    
    print(f"FP16 (no master copy) Memory Usage:")
    print(f"  After model:    {mem_after_model_no_master:.3f} GB")
    print(f"  After forward:  {mem_after_forward_no_master:.3f} GB")
    print(f"  After backward: {mem_after_backward_no_master:.3f} GB")
    print(f"  Peak:           {peak_mem_no_master:.3f} GB")
    
    # Clean up
    del model_no_master, trainer_no_master, logits, loss, token_ids, targets, token_ids_cast
    torch.cuda.empty_cache()
    
    # Test FP16 default (use_autocast=False, use_master_copy=True - default)
    print("\nTesting FP16 (default mixed precision)...")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()
    
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
    
    token_ids, targets = make_data(
        BATCH_SIZE, CONTEXT_LENGTH, model_config["vocab_size"], device=device
    )
    
    # Measure memory after model creation
    mem_after_model_fp16 = torch.cuda.max_memory_allocated() / (10**9)
    
    # Run one forward-backward pass
    (token_ids_cast,) = trainer_fp16.cast_inputs(token_ids)
    with trainer_fp16.forward_context():
        logits = trainer_fp16.model(token_ids_cast)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, model_config["vocab_size"]), targets.view(-1)
    )
    
    # Measure memory after forward
    mem_after_forward_fp16 = torch.cuda.max_memory_allocated() / (10**9)
    
    scaled_loss = trainer_fp16.scale_loss(loss)
    scaled_loss.backward()
    
    # Measure memory after backward
    mem_after_backward_fp16 = torch.cuda.max_memory_allocated() / (10**9)
    
    trainer_fp16.step(check_overflow=False)
    
    # Peak memory
    peak_mem_fp16 = torch.cuda.max_memory_allocated() / (10**9)
    
    print(f"FP16 (default) Memory Usage:")
    print(f"  After model:    {mem_after_model_fp16:.3f} GB")
    print(f"  After forward:  {mem_after_forward_fp16:.3f} GB")
    print(f"  After backward: {mem_after_backward_fp16:.3f} GB")
    print(f"  Peak:           {peak_mem_fp16:.3f} GB")
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"Memory Comparison:")
    print(f"{'='*60}")
    print(f"Peak memory FP32:        {peak_mem_fp32:.3f} GB")
    print(f"Peak memory autocast:    {peak_mem_autocast:.3f} GB")
    print(f"Peak memory no_master:   {peak_mem_no_master:.3f} GB")
    print(f"Peak memory default:     {peak_mem_fp16:.3f} GB")
    
    # Store results
    result_row['peak_mem_fp32_gb'] = peak_mem_fp32
    result_row['peak_mem_autocast_gb'] = peak_mem_autocast
    result_row['peak_mem_no_master_gb'] = peak_mem_no_master
    result_row['peak_mem_default_gb'] = peak_mem_fp16
    
    results_list.append(result_row)
    
    # Clean up
    del model_fp16, trainer_fp16, logits, loss, token_ids, targets, token_ids_cast
    torch.cuda.empty_cache()


if __name__ == "__main__":
    print("Benchmarking Transformer models: FP32 vs FP16 Memory Usage")
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
    csv_filename = os.path.join(output_dir, f"memory_mixprec_benchmark_{timestamp}.csv")
    
    if results:
        fieldnames = [
            'model_name', 'num_layers', 'd_model', 'num_heads', 'd_ff',
            'context_length', 'batch_size',
            'peak_mem_fp32_gb', 'peak_mem_autocast_gb', 
            'peak_mem_no_master_gb', 'peak_mem_default_gb',
        ]
        
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\n{'='*60}")
        print(f"Results exported to: {csv_filename}")
        print(f"{'='*60}")
