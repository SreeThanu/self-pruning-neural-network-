# Self-Pruning ResNet-18 for CIFAR-10

A neural network that learns which of its own neurons and filters to remove during training, using Hard Concrete gates (Louizos et al., ICLR 2018).

## Key Features

- **Hard Concrete Gates**: Differentiable L0 regularization with exact zeros, applied per-channel (structured pruning) on conv layers and per-neuron on FC layers
- **Composite Regularization**: L1 sparsity + entropy penalty pushes gates to decisive 0/1 values
- **Lambda Annealing**: Warmup schedule lets the network learn features before sparsity pressure
- **Rigorous Baselines**: Learned pruning compared against random and magnitude pruning at matched sparsity levels
- **FLOPs Analysis**: Theoretical compute reduction, not just parameter count
- **PrunableResNet18**: Full ResNet-18 adapted for CIFAR-10 with 21 prunable layers

## Architecture

```
PrunableResNet18 (CIFAR-10 variant)
├── Stem: PrunableConv2d(3→64)
├── Layer1: 2× PrunableResBlock(64→64)
├── Layer2: 2× PrunableResBlock(64→128)
├── Layer3: 2× PrunableResBlock(128→256)
├── Layer4: 2× PrunableResBlock(256→512)
├── FC1: PrunableLinear(512→256)
└── FC2: nn.Linear(256→10)
```

Each prunable layer has a learnable gate parameter. During training, gates are sampled stochastically (Hard Concrete distribution). At inference, gates are deterministic — channels/neurons with gate ≈ 0 are effectively removed.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Runs 6 experiments (3 lambda values × 2 schedules), baselines, and generates plots to `results/`.

## Experiments

| Experiment | λ₁ (L1) | Schedule | Description |
|---|---|---|---|
| 1-3 | 1e-4, 5e-4, 1e-3 | Fixed | Constant sparsity pressure |
| 4-6 | 1e-4, 5e-4, 1e-3 | Annealed | Linear warmup over 10 epochs |

## Visualizations

- **Pareto Frontier**: Accuracy vs sparsity for all methods
- **Gate Distributions**: Histogram of gate values per experiment
- **Gate Evolution**: Heatmap showing when gates commit during training
- **Layer-wise Sparsity**: Which layers get pruned most
- **FLOPs vs Accuracy**: Compute efficiency tradeoff
- **Training Curves**: Loss and accuracy over epochs

## Project Structure

```
├── config.py       # Hyperparameters and experiment configs
├── layers.py       # PrunableLinear, PrunableConv2d (Hard Concrete gates)
├── model.py        # PrunableResNet18, PrunableResBlock
├── losses.py       # L1, entropy, and composite sparsity losses
├── train.py        # Training loop, dataloaders, experiment runner
├── evaluate.py     # FLOPs, layerwise sparsity, parameter counts
├── baselines.py    # Random and magnitude pruning baselines
├── visualize.py    # 6 plot functions
├── main.py         # Orchestration entry point
└── tests/          # Unit tests for all components
```

## References

- Louizos, Welling, Kingma. "Learning Sparse Neural Networks through L0 Regularization." ICLR 2018.
- He, Zhang, Ren, Sun. "Deep Residual Learning for Image Recognition." CVPR 2016.
- Frankle, Carlin. "The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks." ICLR 2019.
- Li et al. "Pruning Filters for Efficient ConvNets." ICLR 2017.
- Zhu, Gupta. "To Prune or Not to Prune: Exploring the Efficacy of Pruning for Model Compression." NeurIPS 2017.
- Gale, Elsen, Hooker. "The State of Sparsity in Deep Neural Networks." 2019.
