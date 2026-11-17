# Experiments

Every experiment is a frozen list of *units* (one method configuration × learning rates × orientations × seeds). Workers claim units by atomic file
creation, write one `.npz` per unit (validation, test, hard-half and easy-half curves every 100 epochs, plus the trajectory of the trainable
coefficient) and checkpoint every 2000 epochs, so a run survives hard kills and reboots. Outputs go to `$GRADRESIDUAL_RESULTS/<experiment>/`.

## Scoring

The score at budget $T$ is the mean test MSE over the last 25% of the evaluations up to $T$; orientations are averaged within each seed. A seed has
crashed if this mean is non-finite or ≥ 10. Two rules are available: `drop` (default; crashed seeds are removed and comparisons use the seeds kept on
both sides) and `penalty` (a crashed seed scores 10). Each method is compared at its own best configuration; a difference is called significant when
its paired 95% interval excludes 0.

## `lr_budget` — every method at its own best

| | |
|---|---|
| methods | all of [methods.md](methods.md); 38 configurations per variant: $s,\ s_0,\ c_0 \in \lbrace 0.01,\ 0.1,\ 1/\sqrt{d},\ 1 \rbrace$; fixed $c \in \lbrace 0.5,\ 1,\ 2 \rbrace$; Addition $4 \times 4$ plus $(s, c) = (1, 1)$; $\alpha_0 \in \lbrace -3,\ -1,\ 1,\ 3 \rbrace$ |
brace$, fixed $c \in \lbrace 0.5,\ 1,\ 2 
brace$, Addition $4 	imes 4$ plus $(s, c) = (1, 1)$, $lpha_0 \in \lbrace -3,\ -1,\ 1,\ 3 
brace$) |
| learning rates | 0.5, 0.125, 0.03125, 0.0078125, 0.001953125, 0.00048828125 |
| cells | variants F, D × widths 8, 16, 32 × $f = 1, 2$ × orientations right, left |
| seeds, length | 30 seeds ($\mathrm{seed}_0 = 5000$), 20 000 epochs; budgets 5000 / 10 000 / 20 000 are read from the same curves |
| size | 450 units; about two days on one consumer GPU with several workers |

```bash
gradresidual_train   -e lr_budget --make-manifest
gradresidual_train   -e lr_budget --worker                     # repeat in several terminals
gradresidual_analyze -e lr_budget --freeze                     # selected.json + analysis_drop.json
gradresidual_analyze -e lr_budget --rule penalty               # the same with crashed seeds scored 10
```

`analysis_drop.json` holds the selected configuration of every method in every scenario (with the trained coefficient: median and 10–90% range over
seeds), the tallies against Regular and Standard Residual (overall and hard half), the frequency × budget interaction, the learning-rate × epochs
comparison and the trained-coefficient summary by width.

## `shortcut_scale` — size of the shortcut and training crashes

Standard Residual with the raw shortcut ($s = 1$), the shortcut scaled by a constant $s^{\ast}$ (no per-sample normalisation) and the per-sample
normalised shortcut; variants F, D × widths 8, 16, 32; learning rates 0.5, 0.125, 0.03125, 0.0078125; $f = 1$, right orientation; 100 seeds
($\mathrm{seed}_0 = 3000$), 5000 epochs. `analysis_drop.json` lists the crashed seeds and scores per shortcut and learning rate.

## `shuffled_direction` — does the per-sample direction matter?

Grad Residual with $c = 1$ fixed, with the true $\hat{g}$ and with $\hat{g}$ shuffled across the samples of each forward pass (lengths preserved), against
Regular; variants F, D × widths 8, 16, 32; learning rate 0.0078125; both orientations; 100 seeds ($\mathrm{seed}_0 = 2000$), 5000 epochs.
`analysis_drop.json` gives shuffled − true, shuffled − Regular and true − Regular, paired by seed.

## Sharing a GPU

`scripts/gpu_watchdog.sh` stops the most recently started worker whenever the used GPU memory exceeds `LIMIT_MIB`; the stopped group resumes from its
last checkpoint when a worker is started again (after `gradresidual_train -e <experiment> --release-claims` if no worker is running).
