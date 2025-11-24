from dataclasses import dataclass
import random
import torch

DEFAULT_SEED = 599


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class MemoryStats:
    curr_allocated_bytes: int
    curr_reserved_bytes: int
    max_allocated_bytes: int
    max_reserved_bytes: int


def measure_peak_memory(
    device: torch.device,
    reset_after_measure: bool = True,
) -> MemoryStats:
    torch.cuda.synchronize(device)
    curr_allocated_bytes = torch.cuda.memory_allocated(device)
    curr_reserved_bytes = torch.cuda.memory_reserved(device)
    max_allocated_bytes = torch.cuda.max_memory_allocated(device)
    max_reserved_bytes = torch.cuda.max_memory_reserved(device)

    if reset_after_measure:
        torch.cuda.reset_peak_memory_stats(device)

    return MemoryStats(
        curr_allocated_bytes=curr_allocated_bytes,
        curr_reserved_bytes=curr_reserved_bytes,
        max_allocated_bytes=max_allocated_bytes,
        max_reserved_bytes=max_reserved_bytes,
    )


def start_memory_tracing() -> None:
    torch.cuda.memory._record_memory_history(max_entries=1000000)


def stop_memory_tracing(output_path: str) -> None:
    torch.cuda.memory._dump_snapshot(output_path)
    torch.cuda.memory._record_memory_history(enabled=None)
