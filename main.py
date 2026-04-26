import os
import torch
from config import CONFIG
from train import set_seed, get_dataloaders, run_experiment
from evaluate import compute_layerwise_sparsity, compute_flops, compute_param_counts
from baselines import random_pruning_eval, magnitude_pruning_eval
from visualize import generate_all_plots


def print_summary_table(experiment_results, baseline_results, flops_data):
    header = f"{'Experiment':<25} | {'Accuracy':>8} | {'Sparsity':>8} | {'FLOPs Red.':>10}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for result in experiment_results:
        name = result["name"]
        acc = result["final_test_acc"]
        sp = result["final_sparsity"] * 100
        flop_red = flops_data[name]["reduction_pct"] if name in flops_data else 0.0
        print(f"{name:<25} | {acc:>7.2f}% | {sp:>7.2f}% | {flop_red:>9.1f}%")

    print(sep)

    sparsity_levels = sorted(set(r["final_sparsity"] for r in experiment_results))
    if sparsity_levels:
        mid_sp = sparsity_levels[len(sparsity_levels) // 2]
        if "random" in baseline_results and mid_sp in baseline_results["random"]:
            acc = baseline_results["random"][mid_sp]
            print(f"{'Random Pruning':<25} | {acc:>7.2f}% | {mid_sp*100:>7.2f}% | {'N/A':>10}")
        if "magnitude" in baseline_results and mid_sp in baseline_results["magnitude"]:
            acc = baseline_results["magnitude"][mid_sp]
            print(f"{'Magnitude Pruning':<25} | {acc:>7.2f}% | {mid_sp*100:>7.2f}% | {'N/A':>10}")

    print(sep)


def main():
    set_seed(CONFIG["seed"])
    device = CONFIG["device"]
    print(f"Device: {device}")
    print(f"Running {len(CONFIG['experiments'])} experiments\n")

    trainloader, testloader = get_dataloaders(CONFIG["batch_size"])

    experiment_results = []
    for i, exp_config in enumerate(CONFIG["experiments"]):
        print(f"\n{'='*60}")
        print(f"Experiment {i+1}/{len(CONFIG['experiments'])}: {exp_config['name']}")
        print(f"{'='*60}")
        result = run_experiment(exp_config, trainloader, testloader, device)
        experiment_results.append(result)
        print(f"Final — Acc: {result['final_test_acc']:.2f}%, Sparsity: {result['final_sparsity']*100:.2f}%")

    print(f"\n{'='*60}")
    print("Running baselines...")
    print(f"{'='*60}")

    sparsity_levels = sorted(set(r["final_sparsity"] for r in experiment_results))
    best_model = max(experiment_results, key=lambda r: r["final_test_acc"])["model"]

    print("  Random pruning...")
    random_results = random_pruning_eval(best_model, testloader, sparsity_levels, device)
    print("  Magnitude pruning...")
    magnitude_results = magnitude_pruning_eval(best_model, testloader, sparsity_levels, device)
    baseline_results = {"random": random_results, "magnitude": magnitude_results}

    print(f"\n{'='*60}")
    print("Computing evaluation metrics...")
    print(f"{'='*60}")

    layerwise_data = {}
    flops_data = {}
    for result in experiment_results:
        name = result["name"]
        model = result["model"].cpu()
        layerwise_data[name] = compute_layerwise_sparsity(model)
        flops_data[name] = compute_flops(model, input_size=(3, 32, 32))
        params = compute_param_counts(model)
        print(f"  {name}: {params['effective_params']:,} effective params "
              f"(of {params['total_params']:,} total)")

    print_summary_table(experiment_results, baseline_results, flops_data)

    print(f"\nGenerating visualizations...")
    generate_all_plots(experiment_results, baseline_results, layerwise_data,
                       flops_data, CONFIG["results_dir"])

    print("\nDone!")


if __name__ == "__main__":
    main()
