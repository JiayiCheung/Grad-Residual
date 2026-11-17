# Grad-Residual

Gradient residual connections on a synthetic sine-regression benchmark: a small MLP adds the normalised input-gradient direction of its last
block, `c · ĝ`, inside the last ReLU, next to (or instead of) the usual shortcut `s · x`. This repository trains many such networks in parallel,
selects every method at its own best learning rate and coefficients, and compares them paired by seed.

```
h1 = ReLU(W1 x + b1),  h2 = ReLU(W2 h1 + b2),  z = W3 h2 + b3
h3 = ReLU(z + s · x_b + c · ĝ),   ĝ = g / ‖g‖,   g = ∂ Σ ReLU(z) / ∂ x_b,   y = Wo h3 + bo
```

## Installation

```bash
git clone https://github.com/JiayiCheung/Grad-Residual.git
cd Grad-Residual
pip install -e .            # requires PyTorch >= 2.0; a CUDA GPU is strongly recommended
```

## Usage

```bash
export GRADRESIDUAL_RESULTS=/path/to/results        # default: ./results

gradresidual_train   -e lr_budget --make-manifest    # freeze the list of runs
gradresidual_train   -e lr_budget --worker           # train (start several workers to share one GPU; resumable)
gradresidual_analyze -e lr_budget --freeze           # select each method's best configuration and compare
```

Registered experiments: `lr_budget`, `shortcut_scale`, `shuffled_direction` — see [documentation/experiments.md](documentation/experiments.md).

## Documentation

- [Methods and network](documentation/methods.md)
- [Experiments, protocol and how to reproduce](documentation/experiments.md)

## Tests

```bash
pip install -e ".[test]" && pytest tests
```
