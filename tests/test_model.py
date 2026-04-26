import torch
import pytest
from model import PrunableResNet18


class TestPrunableResNet18:
    def setup_method(self):
        torch.manual_seed(42)
        self.model = PrunableResNet18(num_classes=10, gate_type="hard_concrete")

    def test_output_shape(self):
        x = torch.randn(4, 3, 32, 32)
        out = self.model(x)
        assert out.shape == (4, 10)

    def test_get_all_gate_values_returns_dict(self):
        gates = self.model.get_all_gate_values()
        assert isinstance(gates, dict)
        assert len(gates) > 0

    def test_all_gates_have_correct_range(self):
        gates = self.model.get_all_gate_values()
        for name, g in gates.items():
            assert (g >= 0).all(), f"Gate {name} has negative values"
            assert (g <= 1).all(), f"Gate {name} has values > 1"

    def test_total_sparsity_returns_float(self):
        sparsity = self.model.get_total_sparsity()
        assert isinstance(sparsity, float)
        assert 0.0 <= sparsity <= 1.0

    def test_prunable_layer_count(self):
        gates = self.model.get_all_gate_values()
        # stem(1) + 8 blocks * 2 convs(16) + 3 projections(layer2,3,4) + FC1(1) = 21 prunable layers
        # Exact count depends on which blocks need projection
        assert len(gates) >= 18, f"Expected at least 18 prunable layers, got {len(gates)}"

    def test_gradient_flows_to_all_gates(self):
        x = torch.randn(2, 3, 32, 32)
        out = self.model(x)
        loss = out.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            if "log_alpha" in name:
                assert param.grad is not None, f"No gradient for {name}"

    def test_sigmoid_gate_type(self):
        model = PrunableResNet18(num_classes=10, gate_type="sigmoid")
        x = torch.randn(2, 3, 32, 32)
        out = model(x)
        assert out.shape == (2, 10)

    def test_final_layer_is_plain_linear(self):
        import torch.nn as nn
        assert isinstance(self.model.fc2, nn.Linear), "FC2 must be plain nn.Linear, not prunable"
