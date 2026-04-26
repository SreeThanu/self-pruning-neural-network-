import torch
import torch.nn as nn
from layers import PrunableConv2d, PrunableLinear


def compute_layerwise_sparsity(model, threshold=0.01):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            result[name] = module.get_sparsity(threshold)
    return result


def compute_flops(model, input_size=(3, 32, 32)):
    dense_flops = 0
    pruned_flops = 0

    spatial_sizes = _trace_spatial_sizes(model, input_size)

    for name, module in model.named_modules():
        if isinstance(module, PrunableConv2d):
            conv = module.conv
            c_in = conv.in_channels
            c_out = conv.out_channels
            k = conv.kernel_size[0]
            h_out, w_out = spatial_sizes.get(name, (1, 1))
            layer_dense = 2 * c_in * c_out * k * k * h_out * w_out
            sparsity = module.get_sparsity()
            dense_flops += layer_dense
            pruned_flops += layer_dense * (1 - sparsity)

        elif isinstance(module, PrunableLinear):
            linear = module.linear
            f_in = linear.in_features
            f_out = linear.out_features
            layer_dense = 2 * f_in * f_out
            sparsity = module.get_sparsity()
            dense_flops += layer_dense
            pruned_flops += layer_dense * (1 - sparsity)

    reduction = 100.0 * (1 - pruned_flops / dense_flops) if dense_flops > 0 else 0.0

    return {
        "dense_flops": dense_flops,
        "pruned_flops": pruned_flops,
        "reduction_pct": reduction,
    }


def _trace_spatial_sizes(model, input_size):
    sizes = {}
    hooks = []

    def make_hook(name):
        def hook_fn(module, inp, out):
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                sizes[name] = (out.shape[2], out.shape[3])
        return hook_fn

    for name, module in model.named_modules():
        if isinstance(module, PrunableConv2d):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    with torch.no_grad():
        dummy = torch.randn(1, *input_size)
        model(dummy)

    for h in hooks:
        h.remove()

    return sizes


def compute_param_counts(model):
    total = sum(p.numel() for p in model.parameters())
    gate_params = 0
    pruned_params = 0
    for name, module in model.named_modules():
        if isinstance(module, PrunableConv2d):
            sparsity = module.get_sparsity()
            layer_params = sum(p.numel() for p in module.conv.parameters())
            pruned_params += int(layer_params * sparsity)
            gate_params += module.log_alpha.numel()
        elif isinstance(module, PrunableLinear):
            sparsity = module.get_sparsity()
            layer_params = sum(p.numel() for p in module.linear.parameters())
            pruned_params += int(layer_params * sparsity)
            gate_params += module.log_alpha.numel()

    return {
        "total_params": total,
        "effective_params": total - pruned_params - gate_params,
        "pruned_params": pruned_params,
        "gate_params": gate_params,
    }
