import pandas as pd
import matplotlib.pyplot as plt


def plot_mem_latency_pareto(csv_path: str) -> None:
    model_config_name = csv_path.split("_")[-1]
    df = pd.read_csv(f"{csv_path}.csv")
    x = df["avg_elapse_ms"].values  # latency
    y = df["peak_mem_bytes"].values  # memory
    y = [mem / 1e9 for mem in y]  # convert to GB
    labels = df["benchmark_name"].values
    markers = ["o", "s", "^", "v", "D", "P", "X", "*"]
    assert len(x) <= len(markers), "Not enough markers for the number of points."

    points = list(zip(x, y, labels))
    # Sort by latency
    points_sorted = sorted(points, key=lambda p: p[0])
    pareto = []
    best_y = float("inf")
    for latency, mem, label in points_sorted:
        if mem < best_y:
            pareto.append((latency, mem, label))
            best_y = mem
    px = [p[0] for p in pareto]
    py = [p[1] for p in pareto]

    fig, ax = plt.subplots(figsize=(10, 7))
    # Pareto frontier highlighted
    ax.plot(
        px, py, "--", color="red", linewidth=1, markersize=1, label="Pareto Frontier"
    )
    # All points
    for (latency, mem, label), marker in zip(points, markers):
        ax.scatter(
            latency, mem, s=80, marker=marker, edgecolor="black", alpha=0.9, label=label
        )

    ax.set_xlabel("Average Latency (ms)", fontsize=12)
    ax.set_ylabel("Peak Memory (GB)", fontsize=12)
    ax.set_title(f"Latency-Memory Tradeoff (Transformer-{model_config_name})", fontsize=14)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{csv_path}.png")
    plt.close()


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to the CSV file (without extension) containing benchmark results.",
    )
    args = parser.parse_args()
    plot_mem_latency_pareto(args.csv_path)
