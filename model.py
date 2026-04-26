import torch
import torch.nn as nn
from layers import PrunableConv2d, PrunableLinear


class PrunableResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, gate_type="hard_concrete"):
        super().__init__()
        self.conv1 = PrunableConv2d(in_channels, out_channels, kernel_size=3,
                                     stride=stride, padding=1, bias=False, gate_type=gate_type)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = PrunableConv2d(out_channels, out_channels, kernel_size=3,
                                     stride=1, padding=1, bias=False, gate_type=gate_type)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                PrunableConv2d(in_channels, out_channels, kernel_size=1,
                               stride=stride, bias=False, gate_type=gate_type),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class PrunableResNet18(nn.Module):
    def __init__(self, num_classes=10, gate_type="hard_concrete", dropout=0.3):
        super().__init__()
        self.gate_type = gate_type

        self.stem = PrunableConv2d(3, 64, kernel_size=3, padding=1, bias=False, gate_type=gate_type)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 64, num_blocks=2, stride=1, gate_type=gate_type)
        self.layer2 = self._make_layer(64, 128, num_blocks=2, stride=2, gate_type=gate_type)
        self.layer3 = self._make_layer(128, 256, num_blocks=2, stride=2, gate_type=gate_type)
        self.layer4 = self._make_layer(256, 512, num_blocks=2, stride=2, gate_type=gate_type)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = PrunableLinear(512, 256, gate_type=gate_type)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(256, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride, gate_type):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        current_in = in_channels
        for s in strides:
            layers.append(PrunableResBlock(current_in, out_channels, stride=s, gate_type=gate_type))
            current_in = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.stem(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = torch.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return out

    def get_all_gate_values(self):
        gates = {}
        for name, module in self.named_modules():
            if isinstance(module, (PrunableLinear, PrunableConv2d)):
                gates[name] = module.get_gate_values()
        return gates

    def get_total_sparsity(self, threshold=0.01):
        total_params = 0
        sparse_params = 0
        for module in self.modules():
            if isinstance(module, PrunableConv2d):
                g = module.get_gate_values()
                params_per_channel = module.conv.weight.numel() // g.numel()
                total_params += module.conv.weight.numel()
                sparse_params += int((g < threshold).sum()) * params_per_channel
            elif isinstance(module, PrunableLinear):
                g = module.get_gate_values()
                params_per_neuron = module.linear.weight.numel() // g.numel()
                total_params += module.linear.weight.numel()
                sparse_params += int((g < threshold).sum()) * params_per_neuron
        return sparse_params / total_params if total_params > 0 else 0.0
