import torch
import torch.nn as nn

_HC_EPS = 1e-8
_VALID_GATE_TYPES = {"hard_concrete", "sigmoid"}


class _HardConcreteGateMixin:
    def _map_gate(self, log_alpha):
        s = torch.sigmoid(log_alpha / self.beta)
        s_bar = s * (self.zeta - self.gamma) + self.gamma
        return s_bar.clamp(0, 1)

    def _hard_concrete_sample(self, log_alpha):
        if self.training:
            u = torch.zeros_like(log_alpha).uniform_(_HC_EPS, 1 - _HC_EPS)
            s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / self.beta)
            s_bar = s * (self.zeta - self.gamma) + self.gamma
            # Clamp enables exact zeros/ones (L0 behavior) but kills gradient when s_bar
            # is already saturated — intentional per Louizos et al. 2018
            return s_bar.clamp(0, 1)
        return self._map_gate(log_alpha)

    def _sigmoid_gate(self, log_alpha):
        return torch.sigmoid(log_alpha)

    def _sample_gate(self):
        if self.gate_type == "hard_concrete":
            return self._hard_concrete_sample(self.log_alpha)
        return self._sigmoid_gate(self.log_alpha)

    def get_gate_values(self):
        with torch.no_grad():
            if self.gate_type == "hard_concrete":
                return self._map_gate(self.log_alpha)
            return torch.sigmoid(self.log_alpha)

    def get_sparsity(self, threshold=0.01):
        gates = self.get_gate_values()
        return float((gates < threshold).sum()) / gates.numel()


class PrunableLinear(_HardConcreteGateMixin, nn.Module):
    def __init__(self, in_features, out_features, bias=True, gate_type="hard_concrete",
                 beta=0.66, gamma=-0.1, zeta=1.1):
        super().__init__()
        if gate_type not in _VALID_GATE_TYPES:
            raise ValueError(f"gate_type must be one of {_VALID_GATE_TYPES}, got {gate_type!r}")
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.gate_type = gate_type
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.log_alpha = nn.Parameter(torch.zeros(out_features))

    def forward(self, x):
        return self.linear(x) * self._sample_gate()


class PrunableConv2d(_HardConcreteGateMixin, nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,
                 bias=True, gate_type="hard_concrete", beta=0.66, gamma=-0.1, zeta=1.1):
        super().__init__()
        if gate_type not in _VALID_GATE_TYPES:
            raise ValueError(f"gate_type must be one of {_VALID_GATE_TYPES}, got {gate_type!r}")
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.gate_type = gate_type
        self.beta = beta
        self.gamma = gamma
        self.zeta = zeta
        self.log_alpha = nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        return self.conv(x) * self._sample_gate().view(1, -1, 1, 1)
