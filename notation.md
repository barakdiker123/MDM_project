# Notation map — paper ↔ code

Reference: O. Kost, J. Duník, I. Puncochář, O. Straka, *Unobservable Systems: No
Problem for Noise Identification*, IEEE TAC 71(2):1223–1230, Feb. 2026.

All vectorisation is **column-major** (Fortran order), matching the reference
MATLAB. For a column vector `a`, the Kronecker power `a ⊗ a` equals `vec(a aᵀ)`
because `aᵢaⱼ = aⱼaᵢ`; this is why the residue second moment can be written as a
Kronecker power (paper) or as `vec` of an outer product (code).

## Model (eqs. 1a–1b)

| paper | code |
|-------|------|
| `x_{k+1} = F_k x_k + G_k u_k + E_k w_k` | `StateSpaceModel.F/G/E`, `.simulate` |
| `z_k = H_k x_k + D_k v_k`              | `StateSpaceModel.H/D` |
| `Q = Σ αᵢ B_Q⁽ⁱ⁾`, `R = Σ αᵢ B_R⁽ⁱ⁾` (eq. 9) | `NoiseBasis.BQ/BR`, `.to_QR` |

## Augmented window (eqs. 2–3)

| paper | code |
|-------|------|
| `L`, `L̄ = L−1` | `MDM.L`, `MDM.Lbar` |
| `O_k` (eq. 3a), `Γ_k` (eq. 3e) | `augmented.observability_gamma` → `(O, Gamma)` |
| `script_G_k, script_E_k, script_D_k` (eqs. 3b–3d) | block-diagonals built in `MDM._precompute` |
| `n_E = L̄ n_w + L n_v` | `Sigma_E` dimension in `moments.augmented_cross_cov` |

## Residue (eqs. 4–7, 31–33)

| paper | code |
|-------|------|
| `am(O_k)`, `am(O_k) O_k = 0` (eq. 4) | `mdm.annihilation_matrix` (SVD left-null) |
| `am([O_k, Γ_k script_G_k])` (eq. 33, unknown input) | `MDM(..., unknown_input=True)` |
| `A_k = am(·)[Γ_k, I]` (eq. 7a / 33) | `MDM.A_k[k]` |
| `C_k = blkdiag(script_E_k, script_D_k)` (eq. 7b) | `C_k` in `_precompute`; product cached as `MDM.AC[k] = A_k C_k` |
| residue `𝒵_k = A_k C_k [W_k; V_k]` (eq. 6b) | `res` in `MDM.residue_cov` |

## Covariance relation & parameterisation (eqs. 8–16)

| paper | code |
|-------|------|
| `R_{𝓔²} = Υ_{𝓔²} α` (eqs. 10–12) | `NoiseBasis.Upsilon_E2(L)` |
| unification `Ξ` (eq. 14) | `linalg.upper_tri_selector` / `selector_matrix` |
| replication `Ψ`, `Ψ Ξ = I` (eq. 30) | implicit (we keep unique entries throughout) |
| `R^U_{𝒵²,k} = Ξ_k A_k^{⊗2} C_k^{⊗2} Υ_{𝓔²} α` (eq. 16) | `MDM.script_A[k]` (the per-window regression block) |

## Estimation (eqs. 17–26)

| paper | code |
|-------|------|
| stacked `R_{𝒵²}` (eq. 20a, lhs) | `concatenate(MDM.residue_cov(...))` |
| stacked `script_A` (eq. 20a) | `MDM.script_A_stacked` |
| `# identifiable = rank(script_A)` (Sec. V-C) | `MDM.n_identifiable()` |
| ordinary `α_o = (script_Aᵀ script_A)⁻¹ script_Aᵀ R_{𝒵²}` (eq. 21) | `MDM.fit_ordinary` |
| `R_{η²} = E[𝓔^{⊗4}] − R_{𝓔²}^{⊗2}` (eq. 18) | Gaussian closed form in `moments.unique_cov_block` |
| weighting `P` (eq. 22) | `MDM._assemble_P` (sparse, banded by lag `s < L`) |
| weighted `α_w = (script_Aᵀ P⁻¹ script_A)⁻¹ script_Aᵀ P⁻¹ R_{𝒵²}` (eq. 23) | `MDM.fit_weighted` |
| `COV(α_w) ≈ (script_Aᵀ P⁻¹ script_A)⁻¹` (eq. 24) | `MDM.fit_weighted(return_cov=True)` |

## Implementation notes

* **Annihilation by SVD.** `am(·)` is the left null space from the SVD (the
  paper notes the choice of `am` is non-unique and, for the weighted MDM, its
  effect is negligible — eqs. 27–29).
* **Weighting in residue space.** Rather than forming `E[𝓔^{⊗4}]` symbolically,
  we use the equivalent Gaussian identity
  `Cov(𝒵ₖ^{⊗2}, 𝒵ⱼ^{⊗2}) = (I+K)(Σ_{𝒵ₖ,𝒵ⱼ} ⊗ Σ_{𝒵ₖ,𝒵ⱼ})`,
  with `Σ_{𝒵ₖ,𝒵ⱼ} = (A_k C_k) Σ_{𝓔ₖ,𝓔ⱼ} (A_j C_j)ᵀ`. This gives the same `P`
  as eq. 22 but numerically and without a symbolic toolbox. The cross term is
  nonzero only for `|k−j| < L`, so `P` is block-banded and is solved sparsely.
* **Numerical scale.** Simulation uses a *relative* Cholesky jitter so the
  `10⁻¹⁹`-scale clock covariances (Example A) are reproduced faithfully; a fixed
  absolute jitter would swamp them.
