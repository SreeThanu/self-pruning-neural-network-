import torch
import pytest
from model import PrunableResNet18
from baselines import random_pruning_eval, magnitude_pruning_eval


@pytest.fixture
def model_and_loader():
    torch.manual_seed(42)
    model = PrunableResNet18(num_classes=10)
    dataset = torch.utils.data.TensorDataset(
        torch.randn(64, 3, 32, 32),
        torch.randint(0, 10, (64,)),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)
    return model, loader


class TestRandomPruning:
    def test_returns_dict(self, model_and_loader):
        model, loader = model_and_loader
        result = random_pruning_eval(model, loader, sparsity_levels=[0.3, 0.5], device="cpu")
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_accuracy_is_percentage(self, model_and_loader):
        model, loader = model_and_loader
        result = random_pruning_eval(model, loader, sparsity_levels=[0.5], device="cpu")
        acc = list(result.values())[0]
        assert 0.0 <= acc <= 100.0


class TestMagnitudePruning:
    def test_returns_dict(self, model_and_loader):
        model, loader = model_and_loader
        result = magnitude_pruning_eval(model, loader, sparsity_levels=[0.3, 0.5], device="cpu")
        assert isinstance(result, dict)
        assert len(result) == 2

    def test_accuracy_is_percentage(self, model_and_loader):
        model, loader = model_and_loader
        result = magnitude_pruning_eval(model, loader, sparsity_levels=[0.5], device="cpu")
        acc = list(result.values())[0]
        assert 0.0 <= acc <= 100.0
