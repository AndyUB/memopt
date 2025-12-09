from dataclasses import dataclass
import json
import logging
import numpy as np
import random
import torch

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_SEED = 599


def set_seed(seed: int = DEFAULT_SEED) -> None:
    """Set the random seed for reproducibility.

    Args:
        seed (int): The seed value to set.
    """

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_master_params(
    model: torch.nn.Module,
    device: torch.device,
    dtype: torch.dtype | None = None,
    pin_memory: bool = False,
) -> list[torch.Tensor]:
    """
    Create FP32 master copy of parameters.

    Args:
        model: The model whose parameters are to be copied.
        device: The device to place the master parameters on.
        dtype: The data type for the master parameters. If None, keep the original
            dtype.
        pin_memory: Whether to pin the memory of the master parameters.

    Returns:
        List of FP32 master parameters.
    """
    master_params = []
    for param in model.parameters():
        master_param = param.detach().clone().to(dtype=dtype, device=device)
        if pin_memory:
            master_param = master_param.pin_memory()
        master_param.requires_grad = param.requires_grad
        master_params.append(master_param)
    return master_params


def assert_model_dtype(model: torch.nn.Module, dtype: torch.dtype) -> None:
    """
    Assert that all model parameters are of the specified dtype.

    Args:
        model: The model to check.
        dtype: The expected data type.
    """
    for param in model.parameters():
        if param.dtype != dtype:
            raise ValueError(
                f"Model parameter has dtype {param.dtype}, expected {dtype}."
            )


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


def get_timing_events(num_events: int = 1) -> list[torch.cuda.Event]:
    events = []
    for _ in range(num_events):
        event = torch.cuda.Event(enable_timing=True)
        events.append(event)
    return events


class TrainingTimer:
    """
    Timer for a typical training step:

    ```
    optimizer.zero_grad()
    output = model(input)
    loss = criterion(output, target)
    loss.backward()
    optimizer.step()
    ```
    """

    def __init__(self):
        (
            self.start_event,
            self.end_event,
            self.forward_start_event,
            self.forward_end_event,
            self.backward_start_event,
            self.backward_end_event,
        ) = get_timing_events(num_events=6)
        self.elapses_ms: dict[str, float] = {}

    def get_elapses(self) -> dict[str, float]:
        self.elapses_ms["zero_grad"] = self.start_event.elapsed_time(
            self.forward_start_event
        )
        self.elapses_ms["forward"] = self.forward_start_event.elapsed_time(
            self.forward_end_event
        )
        self.elapses_ms["loss"] = self.forward_end_event.elapsed_time(
            self.backward_start_event
        )
        self.elapses_ms["backward"] = self.backward_start_event.elapsed_time(
            self.backward_end_event
        )
        self.elapses_ms["step"] = self.backward_end_event.elapsed_time(self.end_event)
        self.elapses_ms["total"] = self.start_event.elapsed_time(self.end_event)
        return self.elapses_ms


def compute_elapses_stats(
    elapses_across_iters: list[dict[str, float]],
) -> dict[str, dict[str, float]]:
    """
    Compute mean, standard deviation, and average percentage for each elapse type.

    Args:
        elapses_across_iters (list[dict[str, float]]): List of elapse dictionaries
            from multiple iterations. For each dictionary, keys are elapse types
            (e.g., "forward", "backward") and values are times in milliseconds.
            "total" is a required key in each dictionary.

    Returns:
        dict[str, dict[str, float]]: A dictionary where each key is an elapse type,
            and each value is another dictionary containing "avg", "std", and "pct".
    """
    stats: dict[str, dict[str, float]] = {}
    if len(elapses_across_iters) == 0:
        return stats

    totals = [elapses["total"] for elapses in elapses_across_iters]
    keys = elapses_across_iters[0].keys()
    for key in keys:
        values = [elapses[key] for elapses in elapses_across_iters]
        avg = np.mean(values)
        std = np.std(values)
        pcts = [v / t * 100.0 for v, t in zip(values, totals)]
        avg_pct = np.mean(pcts)
        stats[key] = {
            "avg": avg,
            "std": std,
            "pct": avg_pct,
        }
    return stats


def log_elapses(
    stats: dict[str, dict[str, float]],
    raw_data: list[dict[str, float]],
) -> None:
    """
    Log the elapse statistics.

    Args:
        stats (dict[str, dict[str, float]]): A dictionary where each key is an
            elapse type, and each value is another dictionary containing "avg",
            "std", and "pct".
        raw_data (list[dict[str, float]]): Raw elapse data from multiple iterations.
    """
    logger.info(json.dumps(stats, indent=4))
    logger.info(json.dumps(raw_data, indent=2))
