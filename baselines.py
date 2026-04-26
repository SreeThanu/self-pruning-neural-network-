import torch
import torch.nn as nn
import copy


@torch.no_grad()
def _evaluate_accuracy(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return 100.0 * correct / total


def random_pruning_eval(model, testloader, sparsity_levels, device):
    results = {}
    for target_sparsity in sparsity_levels:
        pruned_model = copy.deepcopy(model).to(device)
        all_weights = []
        weight_refs = []
        for name, param in pruned_model.named_parameters():
            if "log_alpha" not in name and "bn" not in name and param.dim() >= 2:
                all_weights.append(param.data.view(-1))
                weight_refs.append(param.data)

        total_weights = sum(w.numel() for w in all_weights)
        n_prune = int(total_weights * target_sparsity)

        indices = torch.randperm(total_weights)[:n_prune]
        flat_all = torch.cat(all_weights)
        mask = torch.ones_like(flat_all)
        mask[indices] = 0

        offset = 0
        for param_data in weight_refs:
            numel = param_data.numel()
            param_mask = mask[offset:offset + numel].view(param_data.shape)
            param_data.mul_(param_mask)
            offset += numel

        acc = _evaluate_accuracy(pruned_model, testloader, device)
        results[target_sparsity] = acc
    return results


def magnitude_pruning_eval(model, testloader, sparsity_levels, device):
    results = {}
    for target_sparsity in sparsity_levels:
        pruned_model = copy.deepcopy(model).to(device)
        all_weights = []
        weight_refs = []
        for name, param in pruned_model.named_parameters():
            if "log_alpha" not in name and "bn" not in name and param.dim() >= 2:
                all_weights.append(param.data.view(-1))
                weight_refs.append(param.data)

        flat_all = torch.cat(all_weights)
        n_prune = int(flat_all.numel() * target_sparsity)
        threshold = flat_all.abs().kthvalue(max(n_prune, 1)).values.item()

        for param_data in weight_refs:
            mask = (param_data.abs() >= threshold).float()
            param_data.mul_(mask)

        acc = _evaluate_accuracy(pruned_model, testloader, device)
        results[target_sparsity] = acc
    return results
