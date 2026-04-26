import torch
from layers import PrunableLinear, PrunableConv2d


def _collect_gate_values(model):
    """Collect gate values with gradients for loss computation.

    Does NOT use get_gate_values() since that runs under no_grad.
    Instead recomputes the deterministic MAP gate directly.
    """
    gates = []
    for module in model.modules():
        if isinstance(module, (PrunableLinear, PrunableConv2d)):
            if module.gate_type == "hard_concrete":
                s = torch.sigmoid(module.log_alpha / module.beta)
                s_bar = s * (module.zeta - module.gamma) + module.gamma
                g = s_bar.clamp(0, 1)
            else:
                g = torch.sigmoid(module.log_alpha)
            gates.append(g)
    return torch.cat(gates)


def l1_sparsity_loss(model):
    gates = _collect_gate_values(model)
    return gates.mean()


def entropy_loss(model):
    gates = _collect_gate_values(model)
    eps = 1e-8
    g = gates.clamp(eps, 1 - eps)
    h = -g * torch.log(g) - (1 - g) * torch.log(1 - g)
    return h.mean()


def composite_sparsity_loss(ce_loss, model, lambda1, lambda2, anneal_factor=1.0):
    gates = _collect_gate_values(model)
    l1 = gates.mean()
    eps = 1e-8
    g = gates.clamp(eps, 1 - eps)
    ent = (-g * torch.log(g) - (1 - g) * torch.log(1 - g)).mean()
    return ce_loss + anneal_factor * (lambda1 * l1 + lambda2 * ent)
