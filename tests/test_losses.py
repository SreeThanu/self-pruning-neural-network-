import torch
import pytest
from model import PrunableResNet18
from losses import l1_sparsity_loss, entropy_loss, composite_sparsity_loss


class TestL1SparsityLoss:
    def test_returns_scalar(self):
        model = PrunableResNet18(num_classes=10)
        loss = l1_sparsity_loss(model)
        assert loss.dim() == 0

    def test_nonnegative(self):
        model = PrunableResNet18(num_classes=10)
        loss = l1_sparsity_loss(model)
        assert loss.item() >= 0

    def test_zero_gates_give_low_loss(self):
        model = PrunableResNet18(num_classes=10)
        for m in model.modules():
            if hasattr(m, "log_alpha"):
                m.log_alpha.data.fill_(-100.0)
        loss = l1_sparsity_loss(model)
        assert loss.item() < 0.01

    def test_gradient_flows(self):
        model = PrunableResNet18(num_classes=10)
        loss = l1_sparsity_loss(model)
        loss.backward()
        for name, p in model.named_parameters():
            if "log_alpha" in name:
                assert p.grad is not None and p.grad.abs().sum() > 0, f"Zero gradient for {name}"


class TestEntropyLoss:
    def test_returns_scalar(self):
        model = PrunableResNet18(num_classes=10)
        loss = entropy_loss(model)
        assert loss.dim() == 0

    def test_gradient_flows(self):
        model = PrunableResNet18(num_classes=10)
        # Initialize log_alpha away from 0 so the hard-concrete MAP gate stays in (0,1)
        # and the entropy gradient is non-zero. At log_alpha=0 the gate sits at
        # entropy's maximum (zero gradient); at log_alpha>=~1.9 the gate saturates
        # to 1 via clamp, again killing the gradient.
        for m in model.modules():
            if hasattr(m, "log_alpha"):
                m.log_alpha.data.fill_(0.5)
        loss = entropy_loss(model)
        loss.backward()
        for name, p in model.named_parameters():
            if "log_alpha" in name:
                assert p.grad is not None and p.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_decisive_gates_low_entropy(self):
        model = PrunableResNet18(num_classes=10)
        for m in model.modules():
            if hasattr(m, "log_alpha"):
                m.log_alpha.data.fill_(-100.0)
        loss_decisive = entropy_loss(model)
        model2 = PrunableResNet18(num_classes=10)
        loss_indecisive = entropy_loss(model2)
        assert loss_decisive.item() < loss_indecisive.item()


class TestCompositeSparsityLoss:
    def test_combines_losses(self):
        model = PrunableResNet18(num_classes=10)
        ce_loss = torch.tensor(1.0, requires_grad=True)
        total = composite_sparsity_loss(ce_loss, model, lambda1=1e-4, lambda2=5e-3)
        assert total.item() > ce_loss.item()

    def test_zero_lambdas_equal_ce(self):
        model = PrunableResNet18(num_classes=10)
        ce_loss = torch.tensor(2.5, requires_grad=True)
        total = composite_sparsity_loss(ce_loss, model, lambda1=0.0, lambda2=0.0)
        assert torch.allclose(total, ce_loss)

    def test_annealing_factor(self):
        model = PrunableResNet18(num_classes=10)
        ce_loss = torch.tensor(1.0, requires_grad=True)
        full = composite_sparsity_loss(ce_loss, model, lambda1=1e-3, lambda2=5e-3, anneal_factor=1.0)
        half = composite_sparsity_loss(ce_loss, model, lambda1=1e-3, lambda2=5e-3, anneal_factor=0.5)
        assert full.item() > half.item()
