import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from typing import Any

from memopt.model.transformer import Transformer
from memopt.model.config import OOM_TRANSFORMER_CONFIGS, TRAINABLE_TRANSFORMER_CONFIGS


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def get_model_size_to_param_count_map() -> dict[str, int]:
    """
    Returns a mapping from model size names to their parameter counts.
    """

    size_to_count = {}
    configs = {**OOM_TRANSFORMER_CONFIGS, **TRAINABLE_TRANSFORMER_CONFIGS}
    meta_device = torch.device("meta")
    for size_name, config in configs.items():
        model = Transformer(**config, device=meta_device)
        param_count = count_params(model)
        size_to_count[size_name] = param_count
    return size_to_count


def to_df(data: list[dict[str, Any]], filename: str) -> pd.DataFrame:
    """
    Convert the given data to a DataFrame and save it as a CSV file.

    Args:
        data (list[dict[str, Any]]): The data to save.
        filename (str): The name of the output CSV file.
    """
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)
    return df


def plot_ckpt_times(model_dfs: dict[str, pd.DataFrame], filename: str) -> None:
    """
    model_dfs: dict mapping model_size (str) -> df
    filename: str, the name of the output image file

    Each df must have:
      - column 'ckpt_strat'
      - columns 'forward', 'backward', 'step', 'other'
      - optional column 'total' (if missing, will be computed as sum of the four)
    """
    stack_order = ["forward", "step", "other", "backward"]
    phase_colors = {
        "forward": "#1f77b4",
        "step": "#ff7f0e",
        "other": "#2ca02c",
        "backward": "#7A4C8E",
    }
    model_size_to_param_count = get_model_size_to_param_count_map()

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 1.2
    inner_gap = 0.15  # gap between bars *inside* group
    group_gap = 0.6  # extra gap between different model sizes

    seen_labels = set()

    x_positions = []
    x_labels = []

    current_x = 0.0

    group_centers = []
    group_names = []

    order = ["None", "Blockwise", "Attention", "FFN"]
    for model_size, df in model_dfs.items():
        df["ckpt_strat"] = df["ckpt_strat"].where(df["ckpt_strat"].notna(), "None")
        df = df.sort_values("ckpt_strat")

        group_xs = []
        rows = list(df.iterrows())
        rows.sort(key=lambda x: order.index(x[1]["ckpt_strat"]))

        for _, row in rows:
            x = current_x
            bottom = 0.0

            for phase in stack_order:
                value = float(row[phase])
                label = phase if phase not in seen_labels else None
                ax.bar(
                    x,
                    value,
                    width=bar_width,
                    bottom=bottom,
                    label=label,
                    color=phase_colors[phase],
                )
                seen_labels.add(phase)
                bottom += value

            # total text
            total = float(row["total"])
            ax.text(x, bottom, f"{total:.1f}", ha="center", va="bottom", fontsize=8)

            x_positions.append(x)
            x_labels.append(row["ckpt_strat"])
            group_xs.append(x)

            # move to next bar within same group
            current_x += bar_width + inner_gap

        if group_xs:
            group_centers.append(np.mean(group_xs))
            param_count = model_size_to_param_count[model_size] / 1_000_000
            group_names.append(f"Transformer-{model_size}\n({param_count:.1f}M)")

        # gap between groups
        current_x += group_gap

    ax.set_ylabel("Time (ms)")
    ax.set_title("Training Time Breakdown")

    # ckpt_strat labels (inner)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")

    # group labels (model sizes) *below* ckpt labels
    for center, name in zip(group_centers, group_names):
        ax.text(
            center,
            -0.22,  # a bit below tick labels
            name,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
        )

    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))

    plt.tight_layout()
    plt.savefig(filename)


def plot_ckpt_memory(
    stats: dict[str, dict[str, float]],
    filename: str,
    order: list[str] | None = None,
    stat_unit_ratio_to_gb: float = 1e9,
    **defaults,
) -> None:
    """
    Args:
        stats: dict mapping model_size (str) -> dict mapping
            ckpt_strat (str) -> MemoryStats.max_allocated_bytes
        filename: str, the name of the output image file
    """
    if order is None:
        ckpt_order = ["None", "Blockwise", "Attention", "FFN"]
    else:
        ckpt_order = order

    # Fixed colors: same color for the same strategy across all groups
    ckpt_colors = {
        "None": "#5B4FA1",  # muted purple
        "Blockwise": "#1f77b4",  # blue
        "Attention": "#ff7f0e",  # orange
        "FFN": "#2ca02c",  # green
        "All Optimizations": "#ff7f0e",  # orange
        "No Optimizations": "#1f77b4",  # blue
    }
    model_size_to_param_count = get_model_size_to_param_count_map()

    bar_width = defaults.get("bar_width", 2)
    inner_gap = 0.0  # gap between bars *inside* a group
    group_gap = defaults.get("group_gap", 1)  # extra gap between groups

    fig, ax = plt.subplots(
        figsize=(defaults.get("fig_width", 15), defaults.get("fig_height", 5))
    )

    current_x = 0.0

    group_centers = []
    group_names = []

    seen_labels = set()  # for legend

    # stable iteration over model sizes
    for model_size in stats.keys():
        ckpt_to_mem = stats[model_size]
        group_xs = []

        for ckpt_strat in ckpt_order:
            if ckpt_strat not in ckpt_to_mem:
                continue

            mem_bytes = ckpt_to_mem[ckpt_strat]
            mem_gb = mem_bytes / stat_unit_ratio_to_gb  # GB, not GiB

            x = current_x

            label = ckpt_strat if ckpt_strat not in seen_labels else None
            ax.bar(
                x,
                mem_gb,
                width=bar_width,
                color=ckpt_colors.get(ckpt_strat, "#FF0000"),
                label=label,
            )
            seen_labels.add(ckpt_strat)

            # numeric label on top of bar
            ax.text(
                x,
                mem_gb,
                f"{mem_gb:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

            group_xs.append(x)
            current_x += bar_width + inner_gap

        if group_xs:
            group_centers.append(np.mean(group_xs))
            param_count = model_size_to_param_count[model_size] / 1_000_000
            if param_count >= 10000:
                param_count_str = f"{param_count/1000:.1f}B"
            else:
                param_count_str = f"{param_count:.1f}M"
            model_size_str = f"Transformer-\n{model_size}\n({param_count_str})"
            if defaults.get("model_size_format") == "size_only":
                model_size_str = f"{param_count_str}"
            group_names.append(model_size_str)
            current_x += group_gap

    ax.set_ylabel("Peak memory (GB)")
    ax.set_title(
        defaults.get(
            "title",
            "Peak Memory Usage for Different Activation Checkpointing Strategies",
        )
    )

    # No per-bar tick labels
    ax.set_xticks([])
    ax.set_xticklabels([])

    # Model size labels centered under each group
    for center, name in zip(group_centers, group_names):
        ax.text(
            center,
            -0.06,  # adjust if needed
            name,
            ha="center",
            va="top",
            transform=ax.get_xaxis_transform(),
        )

    # Legend for strategies (colors)
    legend_args = {
        "loc": "upper left",
        "bbox_to_anchor": (1.02, 1.0),
    }
    if defaults.get("use_legend_title", True):
        legend_args["title"] = "Checkpoint strategy"
    ax.legend(**legend_args)

    plt.tight_layout(h_pad=2.0)
    plt.savefig(filename)


def load_ckpt_stats_from_csv(
    path: str, cols: list[str] | dict[str, str] | None = None
) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}
    if cols is None:
        cols = ["Blockwise", "Attention", "FFN", "None"]
    if isinstance(cols, list):
        cols = {col: col for col in cols}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_size = row["model_size"].strip()
            ckpt_dict: dict[str, float] = {}

            for col, alias in cols.items():
                raw = row.get(col, "")
                if raw is None:
                    continue
                raw = raw.strip()
                if raw:  # non-empty cell
                    ckpt_dict[alias] = float(raw)

            # only keep models that have at least one value
            if ckpt_dict:
                stats[model_size] = ckpt_dict

    return stats


def load_stats_from_allopts_csv(path: str) -> dict[str, dict[str, float]]:
    stats: dict[str, dict[str, float]] = {}

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model_name = row["model_name"].strip()
            ckpt_strat = row["ckpt_strat"].strip()
            mem = row["peak_allocated_with_offload_gib"].strip()
            if ckpt_strat == "No Optimizations":
                # continue
                mem = float(mem) / (1024**3)  # convert bytes to GiB

            if model_name not in stats:
                stats[model_name] = {}
            stats[model_name][ckpt_strat] = float(mem)  # in GiB

    return stats


if __name__ == "__main__":
    import sys
    import os

    # first arg is dir containing CSVs
    assert len(sys.argv) == 2, "Usage: python benchmark_util.py <results_dir>"
    results_dir = sys.argv[1]

    # model_dfs = {}
    # for file in os.listdir(results_dir):
    #     if file.startswith("latency_ckpt_") and file.endswith(".csv"):
    #         model_size = file[len("latency_ckpt_") : -len(".csv")]
    #         df = pd.read_csv(os.path.join(results_dir, file))
    #         model_dfs[model_size] = df

    # plot_ckpt_times(model_dfs, os.path.join(results_dir, "latency_actckpt.png"))

    for file in os.listdir(results_dir):
        if file == "mem.csv":
            stats = load_stats_from_allopts_csv(os.path.join(results_dir, file))
            for model_size, ckpt_dict in stats.items():
                for ckpt_strat, mem_gib in ckpt_dict.items():
                    # convert GB back to bytes
                    ckpt_dict[ckpt_strat] = mem_gib * 1024**3
            print(stats)
            plot_ckpt_memory(
                stats,
                os.path.join(results_dir, "mem_allopts.png"),
                order=["No Optimizations", "All Optimizations"],
                title="Max Trainable Model & Corresponding Peak Memory",
                bar_width=1,
                fig_width=8,
                group_gap=1,
                model_size_format="size_only",
                use_legend_title=False,
            )
        elif file.startswith("mem") and "allopts" in file and file.endswith(".csv"):
            stats = load_ckpt_stats_from_csv(
                os.path.join(results_dir, file),
                cols={"peak_allocated_with_offload_gb": "All Optimizations"},
            )
            for model_size, ckpt_dict in stats.items():
                for ckpt_strat, mem_gib in ckpt_dict.items():
                    # convert GB back to bytes
                    ckpt_dict[ckpt_strat] = mem_gib * 1024**3
            print(stats)
            plot_ckpt_memory(
                stats,
                os.path.join(results_dir, "mem_allopts.png"),
                order=["All Optimizations"],
                bar_width=2,
                fig_width=8,
            )
        # elif file.startswith("mem") and "ckpt" in file and file.endswith(".csv"):
        #     stats = load_ckpt_stats_from_csv(os.path.join(results_dir, file))
        #     plot_ckpt_memory(
        #         stats,
        #         os.path.join(results_dir, "mem_ckpt.png"),
        #     )
