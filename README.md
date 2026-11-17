# Grad-Residual

Gradient residual connections on a synthetic sine-regression benchmark: a small MLP adds the normalised input-gradient direction
$c\,\hat{g}$ of its last block inside the last ReLU, next to (or instead of) the usual shortcut $s\,x_b$. This repository trains many such
networks in parallel, selects every method at its own best learning rate and coefficients, and compares them paired by seed.

```math
\begin{aligned}
h_1 &= \mathrm{ReLU}\left(W_1 x + b_1\right) \\
h_2 &= \mathrm{ReLU}\left(W_2 h_1 + b_2\right) \\
z   &= W_3 h_2 + b_3 \\
h_3 &= \mathrm{ReLU}\left(z + s\,x_b + c\,\hat{g}\right) \\
\hat{y} &= W_o h_3 + b_o \\[6pt]
g &= \nabla_{x_b} \textstyle\sum_j \mathrm{ReLU}(z_j), \qquad \hat{g} = \dfrac{g}{\lVert g \rVert_2 + \varepsilon}, \qquad x_b \in \{h_1, h_2\}
\end{aligned}
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
