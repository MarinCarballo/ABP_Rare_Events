# Longitudinal and transverse active work

The saved total active work is now accompanied by its x and y components:

\[
W_a=W_x+W_y,\qquad
W_x=v\int_0^T\cos\theta\circ dx,\qquad
W_y=v\int_0^T\sin\theta\circ dy.
\]

Both components cover the **entire trajectory, from 0 to T**. The first
left-to-right crossing of x=0 supplies the route label
\(m=y(\tau_c)\); it does not stop either integral. Production retains the
existing condition `x(T)>0.5`, observation cadence, unbias weights, and
crossing-y bins. The dynamics and original total-work calculation are
unchanged.

## Numerical definition

With unwrapped stored angles, define
\(\bar\theta_k=(\theta_k+\theta_{k+1})/2\). The added observables are

\[
W_x=v\sum_k\cos\bar\theta_k\,(x_{k+1}-x_k),\qquad
W_y=v\sum_k\sin\bar\theta_k\,(y_{k+1}-y_k).
\]

Every integration step contributes, including steps before and after the
first crossing and outside the occupation-histogram spatial filter. The
displacements already include propulsion, potential force, and translational
noise; no additional `dt` multiplies them. The midpoint uses unwrapped
angles even though plots may display angles in `[-π,π]`.

The old total is still accumulated in its original order. Accordingly
`Wx + Wy` agrees with that total up to floating-point rounding; the total
was not replaced by a new summation expression.

## Checking the constant-angle argument

For a genuinely constant angle θ, the sums telescope exactly:

\[
W_a=v[\cos\theta\,(x_T-x_0)+\sin\theta\,(y_T-y_0)].
\]

Thus, **if the compared paths have the same positive net x displacement,
the same v, and zero net y displacement**, the transverse term vanishes and
the largest work among constant-angle paths is the right-pointing case
θ=0, with \(W_a=v\Delta x\). This conclusion is valid under those assumptions.

Here \(\Delta x=x_T-x_0\) is net displacement, not distance traveled or
arclength. For the usual start x₀=−1 and an endpoint exactly at x=1 it is
2, giving `2v` for θ=0. Actual selected production paths only satisfy
`x(T)>0.5`; they do not all end at x=1, and their final y is not fixed to
zero. Those endpoint differences already prevent a universal comparison
based solely on the crossing slice.

Constant propulsion orientation also does not force a straight spatial
path: the potential force and translational noise can curve it or bring y
back toward its starting value. Conversely, a path's passing through m≈0
does not imply that its orientation stayed at θ=0.

For varying θ, returning to the initial y does **not** make Wy vanish.
For example, rising by h with θ=π/2 contributes `vh`; descending by h
with θ=−π/2 contributes another `vh`. The two contributions add to `2vh`
although the net y displacement is zero. Angles can change between those
segments; this example isolates the sign of each work contribution.

The same point follows from an exact finite-sum identity. With
\(a_k=\sin\bar\theta_k\) and n displacement segments,

\[
W_y=v\left[a_{n-1}y_n-a_0y_0
-\sum_{k=1}^{n-1}y_k(a_k-a_{k-1})\right].
\]

Even with y₀=yₙ=0, the final sum need not vanish. There is therefore no
general theorem that the m≈1 slice has smaller work than the m≈0 slice.
The component means can test whether differences come from reduced
longitudinal work, transverse work, endpoint differences, or a combination.

## Mode-based work reference

The notebook uses the user-confirmed reference displacement \(\ell=2.1\):

\[
W_{\rm flat}^{\rm ref}(v)=2.1v,\qquad
W_{{\rm SB},\pm}^{\rm ref}(v)
=W_{\rm flat}^{\rm ref}(v)\cos\theta_{*,\pm}(v)
=2.1v\cos\theta_{*,\pm}(v).
\]

Set these controls in the notebook configuration cell:

```python
FLAT_THEORY_DISPLACEMENT = 2.1
SB_THEORY_THETA_SLICE = "crossing"
```

The angle source is the existing crossing-slice occupation histogram
(`X_SLICES["crossing"]`, normally `-0.05 <= x < 0.05`). Its resolved modal
bins supply \(\theta_{*,\pm}\) separately for each speed and fixed `(D,T)`.
Both measured modal branches are retained; missing modes leave gaps rather
than being replaced by zero or interpolated angles. The selected source and
effective saved-bin bounds accompany the reference data.

All flat-reference panels use the same `2.1v` setting. Normalized references
divide the raw values by the existing **`v**2 * T`** normalization. The
displacement prefactor `2.1` is independent of the exponent `2`; the previous
accidental `v**2.1 * T` denominator in sweep references was corrected.
These controls affect comparison curves only, leaving the dynamics,
production endpoints, work integrals, and measured errors unchanged.

This is a **mode-informed constant-angle reference**. The identity used to
motivate it assumes a constant orientation, net x displacement \(\ell\),
and zero net y displacement. The actual occupation mode is conditioned on
`x(T)>0.5` and the selected spatial slice; it is not conditioned separately
on the work slices near m=0 or m=1 and does not prove an optimal trajectory
angle. Variable orientations, transverse work, and endpoint variation remain
in the measured full-path work. Mode extraction uses heuristic prominence
filters and saved histogram bins; PDF-height standard errors do not supply
angular confidence intervals or uncertainty bands for this reference.

## Saved data and uncertainty

New JLD2 files add the following moments below both
`active_work/biased/` and `active_work/unbiased/`, using the existing shared
`mass` in each crossing-y bin:

| Key | Quantity |
| --- | --- |
| `sum_Wx`, `sum_Wy` | ΣwWx, ΣwWy |
| `sum_Wx2`, `sum_Wy2` | ΣwWx², ΣwWy² |
| `sum_WxWy` | ΣwWxWy |

Biased moments use w=1. Unbiased moments use the same shifted trajectory
weight as total work. The same five keys are retained below
`chain_statistics/chains/<id>/active_work/unbiased/` for uncertainty
estimation. Existing `sum_W`, `sum_W2`, `mass`, and `mass2` keep their
definitions.

For any crossing-y slice, sum its moments first, then divide by its mass.
For example, \(\langle W_x\rangle=\Sigma wW_x/\Sigma w\). The component
path-to-path covariance is

\[
\operatorname{Cov}(W_x,W_y)=
\frac{\sum wW_xW_y}{\sum w}
-\langle W_x\rangle\langle W_y\rangle.
\]

It matters because
\(\operatorname{Var}(W_a)=\operatorname{Var}(W_x)+
\operatorname{Var}(W_y)+2\operatorname{Cov}(W_x,W_y)\).
This describes the spread of individual path work, not the uncertainty of
its estimated mean. Mean errors use the existing delete-one-chain
jackknife on each component ratio. The corresponding covariance of the
means is

\[
\operatorname{Cov}_{\rm JK}(\bar W_x,\bar W_y)=
\frac{C-1}{C}\sum_c
(R^x_{(-c)}-\bar R^x_{(-)})
(R^y_{(-c)}-\bar R^y_{(-)}).
\]

Adding component error bars in quadrature would discard that covariance.
Both components use the existing normalization `v^2*T` when displaying
normalized work. At v=0 normalized work is undefined.

## Using the new outputs

For a trajectory already held in Julia, compute all three values in one scan:

```julia
work = active_work_components(sys)
work.active_work_x  # v * sum cos(theta_mid) * delta_x
work.active_work_y  # v * sum sin(theta_mid) * delta_y
work.active_work    # original full-path total
```

Run production with the updated code to obtain the component moments;
there is no new physical parameter or component-only simulation option.
In `plot_production_errors_work_theta.ipynb`, use the existing configurable
m≈0 and m≈1 crossing-y slices. The total-work figures remain, and component
tables/figures use the same slice selection and endpoint condition.
Notebook tables include `Cxxx_active_work_components_vs_crossing_y.csv`
and `active_work_component_m_slice_summary.csv`; rows label components
`x`, `y`, and `total`. Closure tables check the component sum and the
second-moment/covariance relation against the independently retained total.
Signed component means can be negative and are not clipped to zero.

The Julia CSV exporter writes separate files:

- `active_work_components_vs_crossing_y.csv`: biased/unbiased pooled moments,
  component means, standard deviations, and path-to-path covariance.
- `chain_active_work_components_vs_crossing_y.csv`: unbiased chain moments
  for reconstructing slice ratios and their jackknife errors.

The existing total-work CSV filenames and columns remain unchanged.
Component exports are emitted only when the corresponding moments exist;
missing legacy components are not filled with zeros or inferred from total
work. Old pooled total-work data cannot uniquely determine Wx and Wy.
Thinned saved paths also cannot reproduce the exact full-resolution work
used by production. Metadata states the component definitions and time
interval explicitly.

For the active-force/displacement definition of propulsion work, see
[Chiarantoni et al., Eq. (3)](https://arxiv.org/pdf/2001.08500). The constant-angle
and summation-by-parts statements above follow directly from the displayed
integrals and sums; they do not assume the system or ensemble of that paper.
