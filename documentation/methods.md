# Methods and network

## Network

Three hidden layers of width $d \in \{8, 16, 32\}$; the shortcut and the gradient term are added inside the last ReLU:

```math
\begin{aligned}
h_1 &= \mathrm{ReLU}\left(W_1 x + b_1\right) \\
h_2 &= \mathrm{ReLU}\left(W_2 h_1 + b_2\right) \\
z   &= W_3 h_2 + b_3 \\
h_3 &= \mathrm{ReLU}\left(z + s\,q + c\,\hat{g}\right) \\
\hat{y} &= W_o h_3 + b_o
\end{aligned}
```

| symbol | meaning |
|---|---|
| $x_b$ | shortcut source and the point where the gradient is taken: $h_1$ (variant **F**, spans two layers) or $h_2$ (variant **D**, one layer) |
| $q$ | $x_b$, or the normalised shortcut $x_b / \max(\lVert x_b \rVert_2,\ 10^{-2})$ (`shortcut_norm="unit"`) |
| $g$ | $\nabla_{x_b} \sum_j \mathrm{ReLU}(z_j)$, per sample |
| $\hat{g}$ | $g / (\lVert g \rVert_2 + 10^{-7})$: length 1 (0 where $g = 0$), a constant for the optimiser (no second-order terms) |

## Methods

| method | $s$ | $c$ | notes |
|---|---|---|---|
| `regular` | $0$ | $0$ | identical for F and D, trained once |
| `standard_fixed` | fixed | $0$ | Standard Residual |
| `standard_train` | trainable, start $s_0$ | $0$ | |
| `grad_only` | $0$ | trainable, start $c_0$ | Grad Residual |
| `grad_fixed` | $0$ | fixed | Grad Residual with fixed $c$; optionally $\hat{g}$ shuffled across samples |
| `addition` | fixed | trainable from $c_0$, or fixed at $1$ | both terms |
| `convex` | $1 - \sigma(\alpha)$ | $\sigma(\alpha)$ | $\alpha$ trainable |
| `grad_mag` | — | — | $h_3 = \left[\mathrm{ReLU}(z);\ \lVert g \rVert_2\right]$: only the length of $g$, as one extra feature |

Trainable coefficients use the same learning rate as the weights.

## Training

Plain SGD (momentum 0, constant learning rate), batch 512, mean squared error. Within one process, several coefficient configurations of one method
× several learning rates × two orientations × $S$ seeds are stacked as independent networks on a leading axis
([`StackedNet`](../gradresidual/network/stacked_net.py)); each network has its own learning rate. Data, initial weights and batch orders are drawn
once per seed and shared by every configuration, so all comparisons are paired by seed. On CUDA one epoch is captured as a CUDA graph.

Three independent random streams per seed set $\mathrm{seed}_0$: data $12345 + \mathrm{seed}_0$, initial weights $7\,000\,000 + \mathrm{seed}_0$,
batch order $9\,000\,000 + \mathrm{seed}_0$; the shuffled-$\hat{g}$ control uses its own stream $31337 + \mathrm{seed}_0$.

## Data

```math
\begin{aligned}
\mathrm{easy}(x) &= \sin(0.5\,x) + 0.5\,\sin(2.5\,x) \\
\mathrm{hard}(x; f) &= \sin(2f\,x) + 0.5\,\sin(7f\,x), \qquad x \in [-4\pi,\ 4\pi]
\end{aligned}
```

Orientation `right` puts the easy part on $x < 0$ and the hard part on $x \ge 0$; `left` is the mirrored target. $f = 1$ is the original target and
$f = 2$ doubles the frequency of the hard half. 3000 training points and 1000 validation points with label noise $\mathcal{N}(0,\ 0.1^2)$; the test
set is a noise-free grid of 3000 points.
