# `mdm.neural` — learned noise-covariance estimator

An **isolated** learning layer that sits on top of the analytic MDM. It is kept
in its own subpackage so the core `mdm` package has no dependency on PyTorch.

## Idea

The ordinary MDM compresses a whole trajectory into one sufficient statistic

```
s = sum_k  script_A_k^T  R^U_{Z2,k}        (length n_alpha)
```

and then solves the linear system `S alpha = s` (eq. 21). The network consumes
the **same** `s` and learns a nonlinear map `s -> alpha` over a distribution of
feasible `alpha`. Trained sim2real, it behaves as a learned shrinkage / Bayesian
estimator:

* **lower variance** than MDM when data is scarce (a single short trajectory),
* at the cost of a **bias** that, unlike the MDM's, does not vanish as the test
  trajectory grows.

So the network and the MDM are not rivals: the MDM is the unbiased, asymptotically
optimal estimator; the network trades a little asymptotic correctness for a lot of
small-sample stability. Comparing them honestly requires giving the MDM a fair
data budget (see `examples/ex_mdm_vs_nn.py`).

## Files

| file | role |
|------|------|
| `dataset.py` | sim2real generator: PSD-feasible `alpha` sampling + simulation, returns `(s, alpha)` pairs |
| `network.py` | `CovarianceEstimator` MLP (`n -> 128 -> 128 -> 64 -> n`, ELU, frozen input-normalisation buffers) |
| `losses.py`  | Frobenius training loss (matrix space) + Stein evaluation metric |
| `train.py`   | training loop (Adam + cosine schedule), exports `nn_best.pt` and a flat `nn_weights.mat` |

## Train

```bash
python -m mdm.neural.train --n-train 40000 --n-val 4000 --epochs 200
```

The defaults target the Example-B system (unobservable, unknown input). To train
on a different system, import `train` and pass your own `(model, basis, est, lo, hi)`,
where `[lo, hi]` is the sampling box for `alpha` (chosen so `Q(alpha), R(alpha)`
stay PSD with low rejection).

## Notes

* The training loss is Frobenius, **not** Stein: an unconstrained network can emit
  a non-PSD `Q`/`R` early in training, which makes the Stein loss undefined. Stein
  is used only for evaluation, after PSD-projecting the prediction.
* `nn_weights.mat` lets the MATLAB comparison scripts run the network with a plain
  forward pass (no Deep Learning Toolbox).
