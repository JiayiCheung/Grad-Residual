# Methods and network

## Network

Three hidden layers of width `d` (8, 16 or 32); the shortcut and the gradient term are added inside the last ReLU:

```
h1 = ReLU(W1 x + b1)
h2 = ReLU(W2 h1 + b2)
z  = W3 h2 + b3
h3 = ReLU(z + s · q + c · ĝ)
y  = Wo h3 + bo
```

| symbol | meaning |
|---|---|
| `x_b` | shortcut source and the point where the gradient is taken: `h1` (variant **F**, spans two layers) or `h2` (variant **D**, one layer) |
| `q` | `x_b`, or `x_b / max(‖x_b‖, 1e-2)` for the normalised shortcut (`shortcut_norm="unit"`) |
| `g` | `∂ Σ_j ReLU(z_j) / ∂ x_b`, per sample |
| `ĝ` | `g / (‖g‖ + 1e-7)`, length 1 (0 where `g = 0`); a constant for the optimiser (no second-order terms) |

## Methods

| method | `s` | `c` | notes |
|---|---|---|---|
| `regular` | 0 | 0 | identical for F and D, trained once |
| `standard_fixed` | fixed | 0 | Standard Residual |
| `standard_train` | trainable from `s0` | 0 | |
| `grad_only` | 0 | trainable from `c0` | Grad Residual |
| `grad_fixed` | 0 | fixed | Grad Residual with fixed `c`; optionally `ĝ` shuffled across samples |
| `addition` | fixed | trainable from `c0`, or fixed at 1 | both terms |
| `convex` | `1 − σ(α)` | `σ(α)` | `α` trainable |
| `grad_mag` | — | — | `h3 = [ReLU(z); ‖g‖]`: only the length of `g`, as one extra feature |

Trainable coefficients use the same learning rate as the weights.

## Training

Plain SGD (momentum 0, constant learning rate), batch 512, mean squared error. Within one process, several coefficient configurations of one method
× several learning rates × two orientations × `S` seeds are stacked as independent networks on a leading axis
([`StackedNet`](../gradresidual/network/stacked_net.py)); each network has its own learning rate. Data, initial weights and batch orders are drawn
once per seed and shared by every configuration, so all comparisons are paired by seed. On CUDA one epoch is captured as a CUDA graph.

Three independent random streams per seed set (`seed0`): data `12345 + seed0`, initial weights `7 000 000 + seed0`, batch order `9 000 000 + seed0`;
the shuffled-`ĝ` control uses its own stream `31337 + seed0`.

## Data

`easy(x) = sin(0.5 x) + 0.5 sin(2.5 x)`, `hard(x; f) = sin(2 f x) + 0.5 sin(7 f x)`, on `x ∈ [−4π, 4π]`. Orientation `right` puts the easy part on
`x < 0`; `left` is the mirrored target. 3000 training points and 1000 validation points with label noise `N(0, 0.1²)`; the test set is a
noise-free grid of 3000 points. `f = 2` doubles the frequency of the hard half.
