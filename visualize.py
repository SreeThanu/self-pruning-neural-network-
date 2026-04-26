import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import torch


def plot_pareto_curve(experiment_results, baseline_results, save_dir):
    fig, ax = plt.subplots(figsize=(10, 7))

    sparsities = [r["final_sparsity"] * 100 for r in experiment_results]
    accuracies = [r["final_test_acc"] for r in experiment_results]
    names = [r["name"] for r in experiment_results]
    ax.scatter(sparsities, accuracies, s=100, c="tab:blue", zorder=5, label="Learned Gates (HC)")
    for i, name in enumerate(names):
        ax.annotate(name, (sparsities[i], accuracies[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=7)

    if "random" in baseline_results:
        sp = sorted(baseline_results["random"].keys())
        ac = [baseline_results["random"][k] for k in sp]
        ax.plot([s * 100 for s in sp], ac, "x--", color="tab:red", markersize=8, label="Random Pruning")

    if "magnitude" in baseline_results:
        sp = sorted(baseline_results["magnitude"].keys())
        ac = [baseline_results["magnitude"][k] for k in sp]
        ax.plot([s * 100 for s in sp], ac, "d--", color="tab:green", markersize=8, label="Magnitude Pruning")

    ax.set_xlabel("Sparsity (%)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Accuracy vs Sparsity — Pareto Frontier", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "pareto_curve.png"), dpi=150)
    plt.close()


def plot_gate_distributions(experiment_results, save_dir):
    n = len(experiment_results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, result in enumerate(experiment_results):
        gates = result["model"].get_all_gate_values()
        all_gates = torch.cat(list(gates.values())).cpu().numpy()
        axes[i].hist(all_gates, bins=50, range=(0, 1), color="tab:blue", alpha=0.7, edgecolor="black")
        axes[i].set_title(result["name"], fontsize=10)
        axes[i].set_xlabel("Gate Value")
        axes[i].set_ylabel("Count")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Gate Value Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gate_distributions.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_gate_evolution(experiment_result, save_dir):
    snapshots = experiment_result["gate_snapshots"]
    if not snapshots:
        return

    epochs = sorted(snapshots.keys())
    layer_names = list(snapshots[epochs[0]].keys())

    all_values = []
    for epoch in epochs:
        epoch_gates = torch.cat([snapshots[epoch][ln] for ln in layer_names]).numpy()
        all_values.append(epoch_gates)

    heatmap_data = np.array(all_values).T

    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap_data, aspect="auto", cmap="RdYlBu_r",
                   vmin=0, vmax=1, interpolation="nearest")
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Gate Index", fontsize=12)
    ax.set_title(f"Gate Evolution — {experiment_result['name']}", fontsize=14)
    ax.set_xticks(range(len(epochs)))
    ax.set_xticklabels(epochs)
    plt.colorbar(im, ax=ax, label="Gate Value")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "gate_evolution.png"), dpi=150)
    plt.close()


def plot_layerwise_sparsity(experiment_results, layerwise_data, save_dir):
    fig, ax = plt.subplots(figsize=(14, 6))

    layer_names = list(list(layerwise_data.values())[0].keys())
    short_names = [n.split(".")[-1] if "." in n else n for n in layer_names]
    x = np.arange(len(layer_names))
    width = 0.8 / len(experiment_results)

    for i, result in enumerate(experiment_results):
        name = result["name"]
        if name in layerwise_data:
            values = [layerwise_data[name][ln] * 100 for ln in layer_names]
            ax.bar(x + i * width, values, width, label=name, alpha=0.8)

    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("Sparsity (%)", fontsize=12)
    ax.set_title("Layer-wise Sparsity by Experiment", fontsize=14)
    ax.set_xticks(x + width * len(experiment_results) / 2)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "layerwise_sparsity.png"), dpi=150)
    plt.close()


def plot_flops_vs_accuracy(experiment_results, flops_data, baseline_results, save_dir):
    fig, ax = plt.subplots(figsize=(10, 7))

    for result in experiment_results:
        name = result["name"]
        if name in flops_data:
            mflops = flops_data[name]["pruned_flops"] / 1e6
            acc = result["final_test_acc"]
            ax.scatter(mflops, acc, s=100, zorder=5)
            ax.annotate(name, (mflops, acc), textcoords="offset points",
                        xytext=(5, 5), fontsize=7)

    if experiment_results and flops_data:
        dense_mflops = list(flops_data.values())[0]["dense_flops"] / 1e6
        ax.axvline(x=dense_mflops, color="gray", linestyle="--", alpha=0.5,
                   label=f"Dense ({dense_mflops:.1f} MFLOPs)")

    ax.set_xlabel("FLOPs (MFLOPs)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("FLOPs vs Accuracy", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "flops_vs_accuracy.png"), dpi=150)
    plt.close()


def plot_training_curves(experiment_results, save_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    for result in experiment_results:
        epochs = [h["epoch"] for h in result["history"]]
        losses = [h["train_loss"] for h in result["history"]]
        accs = [h["test_acc"] for h in result["history"]]
        ax1.plot(epochs, losses, label=result["name"], alpha=0.8)
        ax2.plot(epochs, accs, label=result["name"], alpha=0.8)

    ax1.set_ylabel("Training Loss", fontsize=12)
    ax1.set_title("Training Curves", fontsize=14)
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_curves.png"), dpi=150)
    plt.close()


def generate_all_plots(experiment_results, baseline_results, layerwise_data, flops_data, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    plot_pareto_curve(experiment_results, baseline_results, save_dir)
    plot_gate_distributions(experiment_results, save_dir)
    plot_training_curves(experiment_results, save_dir)
    plot_layerwise_sparsity(experiment_results, layerwise_data, save_dir)
    plot_flops_vs_accuracy(experiment_results, flops_data, baseline_results, save_dir)

    best = max(experiment_results, key=lambda r: r["final_test_acc"])
    plot_gate_evolution(best, save_dir)

    print(f"All plots saved to {save_dir}/")
