import torch
import pytest
from layers import PrunableLinear, PrunableConv2d


class TestPrunableLinearHardConcrete:
    def setup_method(self):
        torch.manual_seed(42)
        self.layer = PrunableLinear(16, 8, gate_type="hard_concrete")

    def test_output_shape(self):
        x = torch.randn(4, 16)
        out = self.layer(x)
        assert out.shape == (4, 8)

    def test_gate_values_shape(self):
        gates = self.layer.get_gate_values()
        assert gates.shape == (8,)

    def test_gate_values_range(self):
        gates = self.layer.get_gate_values()
        assert (gates >= 0).all()
        assert (gates <= 1).all()

    def test_sparsity_returns_float(self):
        sparsity = self.layer.get_sparsity()
        assert isinstance(sparsity, float)
        assert 0.0 <= sparsity <= 1.0

    def test_training_vs_eval_determinism(self):
        x = torch.randn(4, 16)
        self.layer.eval()
        out1 = self.layer(x)
        out2 = self.layer(x)
        assert torch.allclose(out1, out2), "Eval mode should be deterministic"

    def test_training_mode_stochastic(self):
        x = torch.randn(4, 16)
        self.layer.train()
        outputs = [self.layer(x) for _ in range(10)]
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Training mode should be stochastic"

    def test_log_alpha_is_parameter(self):
        param_names = [n for n, _ in self.layer.named_parameters()]
        assert "log_alpha" in param_names

    def test_gradient_flows_through_gates(self):
        x = torch.randn(4, 16)
        out = self.layer(x)
        loss = out.sum()
        loss.backward()
        assert self.layer.log_alpha.grad is not None
        assert (self.layer.log_alpha.grad != 0).any()

    def test_invalid_gate_type_raises(self):
        import pytest
        with pytest.raises(ValueError):
            PrunableLinear(16, 8, gate_type="invalid")


class TestPrunableLinearSigmoid:
    def setup_method(self):
        torch.manual_seed(42)
        self.layer = PrunableLinear(16, 8, gate_type="sigmoid")

    def test_output_shape(self):
        x = torch.randn(4, 16)
        out = self.layer(x)
        assert out.shape == (4, 8)

    def test_eval_is_deterministic(self):
        x = torch.randn(4, 16)
        self.layer.eval()
        out1 = self.layer(x)
        out2 = self.layer(x)
        assert torch.allclose(out1, out2)

    def test_gate_values_are_sigmoid(self):
        gates = self.layer.get_gate_values()
        expected = torch.sigmoid(self.layer.log_alpha.data)
        assert torch.allclose(gates, expected)


class TestPrunableConv2dHardConcrete:
    def setup_method(self):
        torch.manual_seed(42)
        self.layer = PrunableConv2d(3, 16, kernel_size=3, padding=1, gate_type="hard_concrete")

    def test_output_shape(self):
        x = torch.randn(4, 3, 32, 32)
        out = self.layer(x)
        assert out.shape == (4, 16, 32, 32)

    def test_gate_per_channel(self):
        gates = self.layer.get_gate_values()
        assert gates.shape == (16,), "One gate per output channel"

    def test_gate_values_range(self):
        gates = self.layer.get_gate_values()
        assert (gates >= 0).all()
        assert (gates <= 1).all()

    def test_structured_pruning(self):
        """When a gate is zero, the entire output channel should be zero."""
        self.layer.log_alpha.data[0] = -100.0
        self.layer.eval()
        x = torch.randn(4, 3, 32, 32)
        out = self.layer(x)
        assert torch.allclose(out[:, 0, :, :], torch.zeros(4, 32, 32), atol=1e-6)

    def test_gradient_flows(self):
        x = torch.randn(4, 3, 32, 32)
        out = self.layer(x)
        loss = out.sum()
        loss.backward()
        assert self.layer.log_alpha.grad is not None
        assert (self.layer.log_alpha.grad != 0).any()

    def test_stride_changes_spatial(self):
        layer = PrunableConv2d(3, 16, kernel_size=3, stride=2, padding=1)
        x = torch.randn(4, 3, 32, 32)
        out = layer(x)
        assert out.shape == (4, 16, 16, 16)

    def test_sparsity(self):
        self.layer.log_alpha.data.fill_(-100.0)
        sparsity = self.layer.get_sparsity()
        assert sparsity == 1.0

    def test_invalid_gate_type_raises(self):
        with pytest.raises(ValueError):
            PrunableConv2d(3, 16, kernel_size=3, gate_type="invalid")
