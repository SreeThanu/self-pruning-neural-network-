import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import torchvision
import torchvision.transforms as transforms
import random
import numpy as np
import copy
from config import CONFIG
from model import PrunableResNet18
from losses import composite_sparsity_loss, l1_sparsity_loss, entropy_loss


def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def get_dataloaders(batch_size, data_dir="./data"):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["cifar10_mean"], CONFIG["cifar10_std"]),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CONFIG["cifar10_mean"], CONFIG["cifar10_std"]),
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_dir, train=True,
                                             download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(root=data_dir, train=False,
                                            download=True, transform=transform_test)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                               shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                              shuffle=False, num_workers=0)
    return trainloader, testloader


def train_one_epoch(model, trainloader, optimizer, lambda1, lambda2, anneal_factor, device):
    model.train()
    total_loss = 0.0
    total_ce = 0.0
    total_l1 = 0.0
    total_entropy = 0.0
    correct = 0
    total = 0
    ce_fn = nn.CrossEntropyLoss()

    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        ce_loss = ce_fn(outputs, targets)
        l1 = l1_sparsity_loss(model)
        ent = entropy_loss(model)
        loss = ce_loss + anneal_factor * (lambda1 * l1 + lambda2 * ent)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_ce += ce_loss.item()
        total_l1 += l1.item()
        total_entropy += ent.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    n_batches = len(trainloader)
    return {
        "train_loss": total_loss / n_batches,
        "ce_loss": total_ce / n_batches,
        "l1_loss": total_l1 / n_batches,
        "entropy_loss": total_entropy / n_batches,
        "train_acc": 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    ce_fn = nn.CrossEntropyLoss()
    total_loss = 0.0

    for inputs, targets in testloader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = ce_fn(outputs, targets)
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    return {
        "test_acc": 100.0 * correct / total,
        "test_loss": total_loss / len(testloader),
    }


def run_experiment(exp_config, trainloader, testloader, device):
    set_seed(CONFIG["seed"])
    model = PrunableResNet18(
        num_classes=10,
        gate_type=CONFIG["gate_type"],
        dropout=CONFIG["dropout"],
    ).to(device)

    initial_weights = copy.deepcopy(model.state_dict())

    optimizer = optim.Adam(model.parameters(), lr=CONFIG["lr"])
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG["epochs"])

    lambda1 = exp_config["lambda1"]
    lambda2 = exp_config["lambda2"]
    annealed = exp_config["annealed"]
    warmup = CONFIG["warmup_epochs"]

    history = []
    gate_snapshots = {}
    best_acc = 0.0
    best_model_state = None

    for epoch in range(CONFIG["epochs"]):
        if annealed and epoch < warmup:
            # anneal_factor=0.0 at epoch=0 intentional: pure CE warmup before sparsity pressure
            anneal_factor = epoch / warmup
        else:
            anneal_factor = 1.0

        train_metrics = train_one_epoch(model, trainloader, optimizer,
                                         lambda1, lambda2, anneal_factor, device)
        test_metrics = evaluate(model, testloader, device)
        scheduler.step()

        if test_metrics["test_acc"] > best_acc:
            best_acc = test_metrics["test_acc"]
            best_model_state = copy.deepcopy(model.state_dict())

        sparsity = model.get_total_sparsity()

        epoch_log = {
            "epoch": epoch + 1,
            **train_metrics,
            **test_metrics,
            "sparsity": sparsity,
            "anneal_factor": anneal_factor,
        }
        history.append(epoch_log)

        if (epoch + 1) % CONFIG["snapshot_interval"] == 0:
            gate_snapshots[epoch + 1] = {
                name: vals.cpu().clone()
                for name, vals in model.get_all_gate_values().items()
            }

        print(f"  Epoch {epoch+1}/{CONFIG['epochs']} | "
              f"Loss: {train_metrics['train_loss']:.4f} | "
              f"Acc: {test_metrics['test_acc']:.2f}% | "
              f"Sparsity: {sparsity:.4f}")

    return {
        "name": exp_config["name"],
        "model": model,
        "initial_weights": initial_weights,
        "history": history,
        "gate_snapshots": gate_snapshots,
        "final_test_acc": history[-1]["test_acc"],
        "final_sparsity": history[-1]["sparsity"],
        "best_test_acc": best_acc,
        "best_model_state": best_model_state,
    }
