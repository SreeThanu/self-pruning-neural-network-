import torch


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


CONFIG = {
    "seed": 42,
    "device": get_device(),
    "batch_size": 128,
    "epochs": 50,
    "lr": 1e-3,
    "dropout": 0.3,
    "gate_type": "hard_concrete",
    "hard_concrete": {
        "beta": 0.66,
        "gamma": -0.1,
        "zeta": 1.1,
    },
    "sparsity_threshold": 0.01,
    "warmup_epochs": 10,
    "snapshot_interval": 5,
    "results_dir": "results",
    "cifar10_mean": (0.4914, 0.4822, 0.4465),
    "cifar10_std": (0.2470, 0.2435, 0.2616),
    "experiments": [
        {"name": "HC_L1=1e-4_fixed",    "lambda1": 1e-4, "lambda2": 5e-3, "annealed": False},
        {"name": "HC_L1=5e-4_fixed",    "lambda1": 5e-4, "lambda2": 5e-3, "annealed": False},
        {"name": "HC_L1=1e-3_fixed",    "lambda1": 1e-3, "lambda2": 5e-3, "annealed": False},
        {"name": "HC_L1=1e-4_annealed", "lambda1": 1e-4, "lambda2": 5e-3, "annealed": True},
        {"name": "HC_L1=5e-4_annealed", "lambda1": 5e-4, "lambda2": 5e-3, "annealed": True},
        {"name": "HC_L1=1e-3_annealed", "lambda1": 1e-3, "lambda2": 5e-3, "annealed": True},
    ],
}
