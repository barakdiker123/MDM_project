# mdm — Measurement Difference Method (Python)

A from-scratch Python re-implementation, in the **notation of the paper**, of the
measurement difference method for noise-covariance identification:

> O. Kost, J. Duník, I. Puncochář, O. Straka,
> **"Unobservable Systems: No Problem for Noise Identification,"**
> *IEEE Transactions on Automatic Control*, 71(2):1223–1230, Feb. 2026.

It mirrors the structure of the authors' MATLAB toolbox and reproduces all three
numerical illustrations. The method identifies the process- and measurement-noise
covariances `Q`, `R` of a linear time-varying state-space model from data, and —
unlike classical correlation methods — works even when the system is
**unobservable** and the **input is unknown**, by cancelling the state with an
*annihilation matrix* instead of a pseudoinverse.

## Why this is interesting

For a window of `L` measurements, the **residue** `𝒵_k` (eqs. 5–6) is a linear
function of the noises alone — the unknown state (and, optionally, the unknown
input) is annihilated. Its covariance is therefore linear in the unknown noise
covariances, so identification reduces to a least-squares problem (eqs. 19–23).
The noise structure is encoded once and for all through *structure-defining
matrices* `Q = Σ αᵢ B_Q⁽ⁱ⁾`, `R = Σ αᵢ B_R⁽ⁱ⁾`, and the method estimates the
weights `α`.

## Install

```bash
pip install -r requirements.txt        # numpy, scipy  (torch optional, for mdm.neural)
pip install -e .                       # editable install of the `mdm` package
```

## Quick start

```python
import numpy as np
from mdm import StateSpaceModel, NoiseBasis, MDM

# ... define F, E, H, D, G_fn, u_fn, and the structure-defining matrices BQ, BR
model = StateSpaceModel.from_constant(F, E, H, D, tau=1000, G_fn=G_fn, u_fn=u_fn)
basis = NoiseBasis(BQ, BR)
est   = MDM(model, basis, L=2, unknown_input=True)

print("identifiable weights:", est.n_identifiable())   # rank(script_A)

z = model.simulate(Q_true, R_true, np.random.default_rng(0))
alpha_o = est.fit_ordinary(z)                # eq. 21  (unbiased, consistent)
alpha_w, cov = est.fit_weighted(z, return_cov=True)   # eq. 23–24 (lower variance)
Q_hat, R_hat = basis.to_QR(alpha_w)
```

## Package layout

```
mdm/
  linalg.py           vec/Kron conventions, commutation, unification (Ξ), PSD tools
  parameterization.py NoiseBasis: structure-defining matrices, Upsilon (eq. 12)
  model.py            StateSpaceModel + simulator (eqs. 1a–1b)
  augmented.py        observability O_k and Gamma_k (eqs. 3a, 3e)
  moments.py          Gaussian fourth-moment weighting blocks (eqs. 18, 22)
  mdm.py              MDM: residue, ordinary (eq. 21) and weighted (eq. 23) LS
  neural/             ISOLATED learned estimator (PyTorch) — see neural/README.md

examples/             ex_A_clock_lti, ex_B_unobservable_unknown_input,
                      ex_C_observable_ltv, ex_mdm_vs_nn
tests/                test_examples (recovers paper alpha), test_weighted (internals)
docs/notation.md      equation-by-equation paper ↔ code map
```

## Reproducing the paper

```bash
python examples/ex_A_clock_lti.py                  # Sec. VII-A (unobservable, LTI clock)
python examples/ex_B_unobservable_unknown_input.py # Sec. VII-B (unobservable, unknown input)
python examples/ex_C_observable_ltv.py             # Sec. VII-C (ordinary vs weighted)
pytest -q                                          # checks identifiability + recovery
```

What you should see:

* **Example B** — ordinary MDM recovers `α = [1, 1, −1, 2, 2, 1]` unbiased
  (6 of 6 identifiable); weighted MDM cuts the RMSE by ~⅓ on a single trajectory.
* **Example C** — both recover `(Q, R) = (2, 1)`; the weighted MDM has the lower
  standard deviation (paper Table III / Fig. 3) and reports a realistic estimate
  covariance via eq. 24.
* **Example A** — the three well-conditioned diffusion weights recover to ~1%;
  the weakly-identifiable white-noise / `R` weights are high-variance, exactly the
  large-STD behaviour of the paper's Fig. 1. (This model spans ~13 orders of
  magnitude — `10⁻¹⁹` noise — so it is numerically delicate; the simulator uses a
  relative Cholesky jitter to handle it.)

## The neural layer

`mdm.neural` is an **isolated** subpackage: a small MLP that maps the MDM
sufficient statistic to `α`, trained sim2real. It is a learned shrinkage estimator
— lower variance than the MDM when data is scarce, at the price of a non-vanishing
bias. See `mdm/neural/README.md` and `examples/ex_mdm_vs_nn.py` for a fair-data
comparison. The core `mdm` package does not depend on PyTorch.

## License of the idea

The method and the three example systems are from Kost et al. (2026), CC-BY 4.0.
This is an independent Python implementation for study and research.
