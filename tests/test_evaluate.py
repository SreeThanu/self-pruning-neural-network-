import torch
import pytest
from model import PrunableResNet18
from evaluate import compute_layerwise_sparsity, compute_flops, compute_param_counts


class TestLayerwiseSparsity:
    def test_returns_dict(self):
        model = PrunableResNet18(num_classes=10)
        result = compute_layerwise_sparsity(model)
        assert isinstance(result, dict)
        assert len(result) > 0

    def test_values_in_range(self):
        model = PrunableResNet18(num_classes=10)
        result = compute_layerwise_sparsity(model)
        for name, sparsity in result.items():
            assert 0.0 <= sparsity <= 1.0, f"{name}: {sparsity}"


class TestFLOPs:
    def test_returns_dict_with_keys(self):
        model = PrunableResNet18(num_classes=10)
        result = compute_flops(model, input_size=(3, 32, 32))
        assert "dense_flops" in result
        assert "pruned_flops" in result
        assert "reduction_pct" in result

    def test_dense_greater_than_pruned(self):
        model = PrunableResNet18(num_classes=10)
        for m in model.modules():
            if hasattr(m, "log_alpha"):
                m.log_alpha.data.fill_(-5.0)
        result = compute_flops(model, input_size=(3, 32, 32))
        assert result["dense_flops"] >= result["pruned_flops"]

    def test_no_pruning_gives_zero_reduction(self):
        model = PrunableResNet18(num_classes=10)
        for m in model.modules():
            if hasattr(m, "log_alpha"):
                m.log_alpha.data.fill_(10.0)
        result = compute_flops(model, input_size=(3, 32, 32))
        assert result["reduction_pct"] < 1.0


class TestParamCounts:
    def test_returns_total_and_effective(self):
        model = PrunableResNet18(num_classes=10)
        result = compute_param_counts(model)
        assert "total_params" in result
        assert "effective_params" in result
        assert result["total_params"] >= result["effective_params"]
