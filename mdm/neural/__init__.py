r"""
mdm.neural -- sim2real neural noise-covariance estimator.

An *isolated* learning layer on top of the analytic MDM. The network consumes
exactly the same 6-dimensional sufficient statistic the (ordinary) MDM uses,

    s = sum_k  script_A_k^T  R^U_{Z2,k},

and maps it nonlinearly to the noise-weight vector alpha. Trained sim2real over
a distribution of feasible alpha, it behaves as a learned shrinkage / Bayesian
estimator: lower variance than MDM in the data-starved single-trajectory regime,
at the price of a bias that does not vanish with more test data.

Modules
-------
dataset  : sim2real generator (PSD-feasible alpha sampling + simulation)
network  : MLP mapping s -> alpha (PyTorch)
losses   : Frobenius training loss + Stein evaluation metric
train    : training loop + checkpoint / weight export

The PyTorch pieces import lazily so the analytic ``mdm`` package has no hard
dependency on torch.
"""
__all__ = ["dataset", "losses"]
