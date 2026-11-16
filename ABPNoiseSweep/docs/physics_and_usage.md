# Physics, algorithms, diagnostics, and usage

This document describes the implemented model and numerical/statistical
conventions. The source code is the authority for results already generated.
The default lambda 2.5 physical model and integration rule are preserved.
The requested lambda 8 comparison is an explicit potential-parameter change.
Optional snapshots do not enter probability estimates.

## 1. Model and potential

The particle position is **r** = `(x,y)` and its direction is
**e**(θ) = `(cos θ, sin θ)`. With unit mobility, the model is

\[
d\mathbf r = [v\mathbf e(\theta)-\nabla U(\mathbf r)]\,dt
             +\sqrt{2D_t}\,d\mathbf B_t,\qquad
d\theta=\sqrt{2D_r}\,dB_t^\theta.
\]

The three Brownian components are independent. Sweeps use `Dt = Dr = D`;
the model object can represent them separately. Setting `potential_active`
false removes the force. Positions have no reflecting boundaries. The
finite bias interval restricts accepted endpoints, not every intermediate
position. Internal angles remain unwrapped.

`src/model.jl` defines the potential below, with configurable λ (default 2.5)
and fixed κ = 0.3:

\[
g(x)=(x^2-1)^2,\quad f_0(y)=\tfrac12\lambda y^2,\quad
f_b(y)=1+\tfrac12\kappa y^2,\quad
f_m(y)=\frac{f_0 f_b}{f_0+f_b},
\]
\[
U(x,y)=g(x)[f_b(y)-f_m(y)]+f_m(y).
\]

Its analytic gradient is

\[
\partial_x U=4x(x^2-1)(f_b-f_m),\qquad
f_m'(y)=\frac{y[\lambda f_b^2+\kappa f_0^2]}{(f_0+f_b)^2},
\]
\[
\partial_y U=g(x)[\kappa y-f_m'(y)]+f_m'(y).
\]

`potential_lambda` selects λ; `ABP_POTENTIAL_LAMBDA` overrides it in the
generic and fixed-D runners. The selected value is used by the potential,
gradient, Hessian, and optional Crank–Nicolson solver, and is saved as
`metadata/potential_lambda`. Keeping λ=2.5 preserves the original force.
Selecting λ=8 changes the physical potential as requested; it is not a
sampling or display setting.

The wells include `(−1,0)` and `(1,0)`. Paths start at `x0_vec` (normally
`[−1,0]`) with θ₀ uniform on `[0,2π)`. The supplied critical speed
`v_c ≈ 0.32` is a user-provided lambda 2.5 reference for choosing the speed
grid. The lambda 8 critical speed may differ; neither its value nor the
transition order is established by these code changes.

## 2. Physical time and integration

There are `N = round(Int,T/dt)+1` stored positions and `N−1` independent
standard-normal three-vectors ξₖ. The actual final grid time is
`T_eff = round(T/dt)*dt`; choose `T/dt` integral when this matters.
The noise scales are σₜ = √(2Dₜdt), σᵣ = √(2Dᵣdt).

The default `integrator = :euler` is the original Euler–Maruyama update:

\[
\mathbf r_{k+1}=\mathbf r_k+
[v\mathbf e(\theta_k)-\nabla U(\mathbf r_k)]\Delta t
+\sigma_T(\xi^x_k,\xi^y_k),\qquad
\theta_{k+1}=\theta_k+\sigma_R\xi^\theta_k.
\]

The optional `:crank_nicolson` uses the same noise and angular update and
trapezoidal position drift **b**(**r**,θ) = v**e**(θ) − ∇U(**r**):

\[
\mathbf r_{k+1}=\mathbf r_k+\frac{\Delta t}{2}
[\mathbf b(\mathbf r_k,\theta_k)+\mathbf b(\mathbf r_{k+1},\theta_{k+1})]
+\sigma_T(\xi^x_k,\xi^y_k).
\]

The implicit step is solved numerically. A failed solve is reported rather
than silently changing the update or resampling the noise. This option
changes the finite-`dt` approximation, not the force, noise strengths, or
specified SDE. It is not a claim of second-order weak accuracy for every
observable. Compare smaller time steps before interpreting differences.
The nonconvex potential can also allow multiple implicit roots at coarse
time steps; solver convergence alone is not a time-step convergence test.
MUCA and production use the same chosen integrator. Acceptance remains in
Gaussian noise space; no coordinate-space Jacobian is inserted into those
noise proposals.

Use `ABP_INTEGRATOR=euler` or `ABP_INTEGRATOR=crank_nicolson`; the generic
and fixed-D runners also accept `--integrator=crank_nicolson`. An explicit
integrator command-line setting takes precedence over the environment.

## 3. A Markov state is a complete trajectory

One MCMC state consists of θ₀ and all ξₖ, which determine a **complete
physical trajectory**. Physical time and the MCMC proposal count are
different axes. A movie of MCMC evolution shows successive whole paths,
not a single particle advancing through physical time.

The unbiased noise-space density is

\[
\pi_0(\theta_0,\{\xi_k\})\propto
\mathbf1_{[0,2\pi)}(\theta_0)
\exp\!\left[-\tfrac12\sum_k\|\xi_k\|^2\right].
\]

The bias coordinate is q = x(T). With binwise logweights ℓ(q), the intended
fixed-bias density inside the endpoint domain is πᵦ ∝ π₀ exp ℓ(q).
Move weights are ordered `[reflection, theta0, block, local]`, defaulting
to `[0, 0.05, 0.55, 0.40]`; only nonnegativity and a positive sum are needed.

`noise_proposal = :random_walk` remains the default. This selects the
existing additive proposals below. An optional preconditioned
Crank–Nicolson **noise proposal**, `:pcn`, is a different setting from the
Crank–Nicolson **physical-time integrator** in section 2.

| Move | Proposal |
| --- | --- |
| Local noise | Select one ξₖ and add `local_dxi * η`, η standard Gaussian; reintegrate the suffix |
| Block noise | Select a start and a requested length from `50,100,250,500,1000`, truncate at the end, add independent `block_dxi * η` vectors; reintegrate the suffix |
| Initial angle | Redraw θ₀ uniformly and reintegrate the complete path |
| Reflection | y → −y, θ → 2π−θ, ξʸ → −ξʸ, ξᶿ → −ξᶿ; default weight zero |

Local/block proposals first pass a Gaussian-prior Metropolis filter:

\[
\alpha_0=\min[1,\exp(-\tfrac12\Delta\sum_k\|\xi_k\|^2)].
\]

On passing it, the trajectory is integrated and the bias filter is
αᵦ = min[1, exp(ℓ(q_new)−ℓ(q_old))]. This two-stage scheme has the
detailed-balance ratio of the product density; its total acceptance
probability generally differs from a single combined Metropolis test.
Outside-domain proposals are rejected. The range is
`first(bins) ≤ q < last(bins)`. All rejections keep the old full trajectory.
The existing domain check tests both old and new endpoints: a chain
initialized outside the bias interval cannot reenter through these moves.
Check initial endpoints and zero-reweight/stuck-chain diagnostics; finite
binning is a sampling restriction, not merely an export choice.

**Existing bookkeeping caveat:** the inspected `MonteCarloX.accept!` records
a histogram visit and increments `alg.steps`. The project calls it only
after the prior/domain filters, then also increments `alg.steps` itself.
Prior/domain rejects are therefore absent from the MUCA learning histogram,
and printed `acceptance_rate` is not accepted proposals divided by all
proposed moves. This can affect the learned bias and must not be read as a
validated learning histogram or convergence certificate. These existing
choices are preserved under the instruction not to change the learning
policy. Production histograms, in contrast, observe the current state after
**every** proposal, including rejections.

For the optional `ABP_NOISE_PROPOSAL=pcn`, a chosen local vector or each
vector in a selected block instead uses

\[
\xi'_k=\sqrt{1-\beta^2}\,\xi_k+\beta\eta_k,
\qquad\eta_k\sim\mathcal N(0,I),\quad 0<\beta\le1.
\]

This proposal is reversible with respect to the standard Gaussian prior,
so the prior/proposal ratio cancels and only the endpoint-bias filter is
needed, following the Gaussian-reference-preserving construction of
[Cotter et al., *MCMC Methods for Functions*](https://arxiv.org/abs/1202.0709).
`ABP_BLOCK_PCN_BETA` and `ABP_LOCAL_PCN_BETA` set β independently
of the existing random-walk amplitudes. With Euler integration this changes
the sampling kernel, while retaining the same discretized physical target.
It does not apply an implicit time step. Default random-walk behavior,
including the documented legacy visit bookkeeping, is preserved. pCN
also does not by itself certify route mixing or learning convergence.

## 4. MUCA learning and histogram flatness

There is one learning chain per Julia thread. In iteration i of I, each
chain thermalizes for `n_therm_muca` moves, resets iteration diagnostics,
and samples approximately

\[
n_i=\max\!\left(1,\operatorname{round}
\frac{\texttt{n_iter_steps_per_iter}\,i}{I\,C}\right)
\]

moves, with C chains. Their merged histogram updates the shared bias.
Thus the final iteration has about `n_iter_steps_per_iter` sampling moves
**across chains**. If enabled, the iteration-count rule is

\[
I(D)=\max(1,\operatorname{round}[
\texttt{n_iter}\sqrt{\texttt{D_scaling_reference}/D}]).
\]

The fixed-noise speed runner disables this scaling.

The code requests `MonteCarloX`'s `:recursive` update. In the inspected
implementation, for adjacent populated bins, dₖ = ℓₖ₊₁−ℓₖ and

\[
\widehat d_k=(\ell_{k+1}-\log H_{k+1})-(\ell_k-\log H_k),\quad
a_k=\frac{H_kH_{k+1}}{H_k+H_{k+1}}.
\]

The new slope estimate is averaged with its previous estimate using
accumulated precision aₖ. Unpopulated pairs are skipped. Integrating slopes
gives the next logweights. Outside the anchors
`xT_min − xT_extension_margin` and `xT_max + xT_extension_margin`, tails
are overwritten with slope `−1/tail_temperature`: temperature 1 for
`D ≥ 0.05`, otherwise 2. This controls bias extension, not physical noise.

For bin centers in `[xT_min,xT_max]`, the saved flatness values are

\[
F_{\max/\mathrm{mean}}=\frac{\max H}{\operatorname{mean}H},\qquad
F_{\mathrm{mean}/\min}=\frac{\operatorname{mean}H}{\min H}.
\]

**Only strictly positive bins enter these ratios.** Empty bins are ignored;
an empty interval returns `Inf`. One visited bin gives both ratios 1
despite poor coverage. These formulas were checked against installed
dependency source; recheck them after an upgrade.

`flatness_threshold` is metadata and does **not** control stopping.
Stopping requires the mean count per chain to exceed
`roundtrip_avg_target_fraction * C` for `roundtrip_convergence_hits`
consecutive iterations. Counts reset each MUCA iteration. In the inspected
dependency each arrival at the opposite boundary after one boundary was
visited adds a count. A→B→A therefore contributes **two counts**: these
are boundary traversals, although output calls them “roundtrips”. Changing
thread count changes this stopping target as well as parallelism.
`roundtrip_target` scales a diagnostic steps-per-target estimate; it is
not the early-stop threshold. The auxiliary tail-stability estimate is
also diagnostic, not a stopping requirement.

Inspect occupied-bin coverage, both flatness ratios, traversals,
independent chains, and reweighted-observable stability together. Present
diagnostics do not establish that all trajectory-space routes have mixed.

## 5. Production and unbiasing

Production creates independent trajectories and reseeded algorithm copies,
thermalizes them, and holds the learned bias fixed. `n_prod_obs_total`
is a total proposal count divided almost equally among `n_prod_chains`;
serial mode uses one chain. Endpoint histograms observe every proposal.

With ℓ* the minimum finite learned logweight, the stored weight is

\[
w_j=\exp[-(\ell(q_j)-\ell_*)].
\]

Out-of-domain or nonfinite-logweight states get zero weight. The common
shift cancels from normalized quantities. An observable is estimated as
ΣwⱼAⱼ / Σwⱼ. `unbiased` arrays are weighted masses, not integer counts;
`biased` arrays use unit weights and describe the biased simulation.

Because endpoints are restricted to the finite bias interval, “all
endpoints” means that simulated domain, not unobserved endpoint tails.
Endpoint windows are `−1 ≤ x(T) < 0`, `x(T)>0`, `x(T)>0.5`, and all; they
overlap deliberately. Observables include y(T), the arithmetic mean of all
stored y points, and the existing discrete sum

\[
\texttt{path_y_int}=\Delta t\sum_{k=0}^{N-1}y_k.
\]

This includes both endpoints and is not trapezoidal quadrature. For
constant y it gives `(T_eff+dt)*y`. That existing convention is preserved.

## 6. Histograms, clipping, and theta

The project's histogram helpers use left-closed/right-open bins with the
final edge included. Outside-range values are dropped and overflow counters
record their number. Normalization uses retained mass:

\[
p_i=\frac{M_i}{(\sum_jM_j)\Delta e_i},\qquad
p_{ij}=\frac{M_{ij}}{(\sum_{kl}M_{kl})\Delta x_i\Delta z_j}.
\]

Zero total mass gives `NaN`. These PDFs integrate to one **inside saved
ranges**. Check overflow before treating a clipped PDF as complete. Path-y
and x-y maps can retain different sets if x is outside the joint x grid.

Path occupation is measured every `path_observation_stride` production
moves, using every `path_time_stride`-th physical-time point. The initial
point is included if it passes the filter `x(t) > path_filter_x_min`.
Endpoint conditions are all, `x(T)>0`, and `x(T)>0.5`. The same
endpoint-derived weight multiplies every selected point on a given path.
A path therefore contributes more retained occupation mass if it spends
more time inside the filter; paths are not individually normalized first.

The orientation histogram is stored only for `x(T)>0.5`. Each selected
point uses `θ_wrapped = mod2pi(θ)` and increments a joint `(x,θ)` array,
with equal angular bins over `[0,2π]`. Its interpretation is

\[
P_{\rm occ}(x(t),\theta(t)\bmod2\pi\mid
x(T)>0.5,\ x(t)>x_{\rm filter},\ \text{retained grid ranges}).
\]

It is **not endpoint theta**. Near 0 and 2π are adjacent directions on a
circle. Angular density is in radian⁻¹, or length⁻¹ radian⁻¹ for the joint
map. The x-theta map has no y clipping, so its retained population can
differ from the x-y map.

The error notebook reads chain matrices using the saved shape and Julia
column-major order (`reshape(..., order="F")`). For an x slice it selects
whole bins by centers, sums **masses**, then normalizes by angular/y bin
widths. Reported effective x limits describe those whole bins; an arbitrary
slice is not interpolated through an existing bin.

## 7. Active work and the “flat” route

The quantity is mechanical active-force work at unit mobility:

\[
W_a=v\int_0^T\mathbf e(\theta)\circ d\mathbf r,\qquad
W_a^{\Delta t}=v\sum_{k=0}^{N-2}
\mathbf e\!\left(\frac{\theta_k+\theta_{k+1}}2\right)
\cdot(\mathbf r_{k+1}-\mathbf r_k).
\]

It uses **unwrapped** midpoint angles and the actual displacement, including
force drift and translational noise. There is no extra `dt`: displacement
already contains it. Work is over the **whole path**, not only its crossing
segment. Normalization is `W_a/(v^2 * trajectory_T)`, undefined at v = 0.
It is not heat, potential work, or work divided by D. The force-work reading
matches the model's unit-mobility convention; thermodynamic interpretation
depends on the active dynamics, as discussed by
[Speck, *Stochastic thermodynamics for active matter*](https://arxiv.org/abs/1603.03195).

The standalone observable and fused occupation/work accumulator implement
the same formula. Work always uses **every integration step**, independently
of histogram time strides and snapshot thinning.

Production measures work every `active_work_observation_stride` proposals
among paths with `x(T)>0.5`. This stride must be a multiple of
`path_observation_stride`. The first segment with xₖ < 0 ≤ xₖ₊₁ sets
α = −xₖ/(xₖ₊₁−xₖ), m = yₖ+α(yₖ₊₁−yₖ), and τ_c = (k+α)dt, with
k zero-based here. Thus m is the interpolated y of the **first left-to-right**
x=0 crossing. No-crossing paths and crossings outside the saved y range
are diagnosed and omitted from route summaries.

For each crossing-y bin production stores Σw, Σw², ΣwWₐ, ΣwWₐ², and
Σwτ_c, pooled and by chain (pooled unit-weight versions are also saved).
Route cuts can be chosen afterward, but within-bin crossing coordinates
and the complete work distribution cannot be recovered from these moments.

The notebook `M_CUT` controls these crossing-y **bin-center** masks:

| Label | Criterion |
| --- | --- |
| `flat` | `abs(m_center) ≤ M_CUT` |
| `upper_sb` | `m_center > M_CUT` |
| `lower_sb` | `m_center < −M_CUT` |
| `combined_sb` | `abs(m_center) > M_CUT` |

This “flat” means a crossing route, not histogram flatness. It does not
require the whole path to stay near y=0 or have low curvature. Configuration
is now the single source of `M_CUT`; the previous later `M_CUT = 1` override
was removed. A cut through a saved bin is resolved by its center. Route means
are ratios of summed weighted work to summed weights. Fractions normalize
among recorded in-range crossing routes, not all generated paths.

## 8. Errors and limits of inference

Reported overall/conditional effective sample sizes are

\[
N_{\rm eff,weights}=\frac{(\sum_jw_j)^2}{\sum_jw_j^2}.
\]

These measure weight concentration and do not correct MCMC autocorrelation.
Points on one physical path are correlated, too; retained time points are
not independent trajectory samples.

The error notebook uses **delete-one-chain jackknife**. For a ratio
R = ΣN_c / ΣZ_c from C independent production chains,

\[
R_{(-c)}=\frac{\sum_dN_d-N_c}{\sum_dZ_d-Z_c},\quad
\overline R_{(-)}=C^{-1}\sum_cR_{(-c)},\quad
\mathrm{SE}_{\rm JK}=\sqrt{\frac{C-1}{C}
\sum_c(R_{(-c)}-\overline R_{(-)})^2}.
\]

For a PDF the bin widths enter Z; for work N is weighted work and Z is
weight. Slices are pooled at mass level before normalization/jackknifing.
One chain, missing arrays, empty bins, or invalid leave-one denominators
do not yield invented precision: errors are `NaN` or that section is skipped.
`BAND_SIGMA` scales display bands; CSV `chain_jackknife_se` remains one SE.
The weighted work standard deviation √(⟨W²⟩−⟨W⟩²) describes path-to-path
spread, not uncertainty of the mean. Multiplying by 1.96 gives an
approximate pointwise band, not simultaneous coverage of a whole curve.

Between-chain variation can reflect correlated sampling if chains are
long, equilibrated, and explore the relevant routes. All chains share the
**same learned bias**: these errors omit learning uncertainty, shared
trapping, integration error, histogram discretization/clipping, and model
uncertainty. Independent complete learning/production runs and time-step
checks address different questions. This maintenance work does not claim
those physical convergence studies were performed.

## 9. Entry points and cluster use

Run from the project root:

```bash
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
julia --project=. --threads=4 scripts/run_noise_sweep.jl --smoke --dry-run
```

Use Julia 1.10 or newer. `Project.toml` pins `MonteCarloX` 0.2 source at
revision `2b2805f4467b2f09074c636faa14abd34c2ce000`: version 0.3 removes
`ImportanceSampling`, required by this code. If an older local manifest
still selects an incompatible release, repair that environment explicitly:

```bash
julia --project=. -e 'using Pkg; Pkg.add(PackageSpec(name="MonteCarloX", url="https://github.com/zierenberg/MonteCarloX.jl", rev="2b2805f4467b2f09074c636faa14abd34c2ce000")); Pkg.instantiate()'
```

Keep the resolved manifest/revision with archived runs; upstream API changes
must not silently change the diagnostic definitions.

`--dry-run` prints resolved settings without sampling. `--smoke` lowers
counts but still integrates many full paths. Environment overrides follow
smoke defaults. `--no-csv` disables CSV unless an environment override
enables it. Plot flags remain for compatibility; Julia execution is data-only.

A deliberately tiny Bash execution/output check is:

```bash
ABP_D_VALUES=0.025 ABP_TRAJECTORY_T=0.1 \
ABP_SCALE_N_ITER_WITH_D=false ABP_N_ITER=1 \
ABP_N_ITER_STEPS_PER_ITER=40 ABP_N_THERM_MUCA=0 \
ABP_N_PROD_OBS_TOTAL=40 ABP_N_PROD_CHAINS=2 ABP_N_THERM_PROD=0 \
ABP_PATH_OBSERVATION_STRIDE=10 ABP_ACTIVE_WORK_OBSERVATION_STRIDE=10 \
ABP_OUTPUT_DIR=data/tiny_check \
julia --project=. --threads=2 scripts/run_noise_sweep.jl
```

It cannot establish a physical distribution or convergence. In PowerShell,
assign environment variables first, e.g.
`$env:ABP_INTEGRATOR = 'crank_nicolson'`, then launch Julia. Clear them later
if they should not affect subsequent runs.

For the speed study, `sbatch run/submit_fixed_D_speed_landscape.sh` runs
23 tasks (`0–22`) for `v=0.280:0.005:0.390`, keeping `Dt=Dr=0.025`.
Each task executes MUCA and production in the same process. Collected results
are under `data/abp_speed_D0p025_array_JOBID/combined/`. Directly launching
the fixed-D Julia runner requires `ABP_V`. Cluster wrappers retain large
research workloads: inspect defaults and cluster paths before submission.

For the new lambda comparison, submit the two dedicated arrays from the
project root:

```bash
mkdir -p errors
sbatch run/submit_production_work_theta_lambda2p5.sh
sbatch run/submit_production_work_theta_lambda8.sh
```

Both arrays have 29 tasks (`0–28`) with the same speeds:
`0.200, 0.240, 0.260`, the fine grid `0.280:0.005:0.390`, then
`0.400, 0.450, 0.500`. Relative to the supplied 0.32 reference, the endpoints
are 37.5% below and 56.25% above. Each fixes `D=0.025`, `T=40`, `dt=0.01`,
Euler integration, and random-walk proposals, with the existing long-run
budgets, 128 CPUs, 96 GB, and a seven-day Slurm limit. The lambda choice,
job/log tags, and output namespace distinguish the two submissions. The
previous submission scripts remain available with their existing grids.

Deploy the updated `src/` and `scripts/run_fixed_D_speed_landscape.jl`
together with these shell files: an older runner/backend ignores the new
lambda variable and would retain 2.5 even for the lambda 8 submission.
The wrappers fix their designated lambda rather than inheriting it from a
previous shell environment. They accept `--dry-run` and the optional
`--integrator=crank_nicolson` argument; unsupported arguments fail clearly.
For example:

```bash
sbatch run/submit_production_work_theta_lambda8.sh --integrator=crank_nicolson
```

For a Bash configuration preflight on a machine with Julia on `PATH`, use:

```bash
for script in run/submit_production_work_theta_lambda2p5.sh run/submit_production_work_theta_lambda8.sh; do
  ABP_PROJECT_DIR="$PWD" JULIA_CMD=julia SLURM_ARRAY_JOB_ID=preflight \
  SLURM_ARRAY_TASK_ID=0 SLURM_CPUS_PER_TASK=1 ABP_N_PROD_CHAINS=1 \
  bash "$script" --dry-run
done
```

This prints one task's configuration per lambda and exits before simulation
and result collection; it can create the output directories. Normal
submissions default to the configured cluster project and Julia executable.
Cluster execution, runtime, and completion within seven days have not been
verified; no job was submitted by this maintenance work.

The separate result roots are
`data/abp_work_theta_lambda2p5_D0p025_array_JOBID/combined/` and
`data/abp_work_theta_lambda8_D0p025_array_JOBID/combined/`. Current notebooks
group by `(D,v,T)`, so set `RUN_ROOTS` to **one lambda at a time** to avoid
combining different potentials under the same group.

For the optional integration rule or complete stored example paths:

```bash
sbatch --export=ALL,ABP_INTEGRATOR=crank_nicolson \
  run/submit_fixed_D_speed_landscape.sh
sbatch --export=ALL,ABP_SAVED_PATH_TIME_THIN=1,ABP_MAX_SAVED_PATHS_PER_WINDOW=300 \
  run/submit_fixed_D_speed_landscape.sh
```

Generic defaults live in `ABPNoiseSweepConfig` and `src/cli.jl`; the fixed-D
runner has dedicated workload defaults. Do not assume they are identical.

| Environment setting | Meaning |
| --- | --- |
| `ABP_D_VALUES` | Generic sweep's comma-separated noise values |
| `ABP_POTENTIAL_LAMBDA` | Potential λ, default 2.5; fixed to 2.5 or 8 by the dedicated lambda submission files |
| `ABP_TRAJECTORY_T`, `ABP_DT`, `ABP_INTEGRATOR` | Physical duration, time step, and integration choice (both runners) |
| `ABP_N_ITER`, `ABP_N_ITER_STEPS_PER_ITER`, `ABP_N_THERM_MUCA` | Learning schedule and per-chain thermalization |
| `ABP_SCALE_N_ITER_WITH_D`, `ABP_D_SCALING_REFERENCE` | Optional iteration-count scaling |
| `ABP_MOVE_WEIGHTS` | Four numbers, or JSON numeric-array file path |
| `ABP_BLOCK_DXI`, `ABP_LOCAL_DXI` | Standardized-noise proposal amplitudes |
| `ABP_NOISE_PROPOSAL` | `random_walk` (default) or Gaussian-prior-preserving `pcn` |
| `ABP_BLOCK_PCN_BETA`, `ABP_LOCAL_PCN_BETA` | pCN proposal amplitudes, each in `(0,1]` |
| `ABP_N_PROD_OBS_TOTAL`, `ABP_N_PROD_CHAINS`, `ABP_N_THERM_PROD` | Production workload |
| `ABP_PATH_OBSERVATION_STRIDE`, `ABP_PATH_TIME_STRIDE` | Occupation cadence and physical-time thinning |
| `ABP_ACTIVE_WORK_OBSERVATION_STRIDE` | Full-resolution work observation cadence |
| `ABP_PATH_FILTER_X_MIN`, `ABP_N_PATH_THETA_BINS` | Retained path region and angular resolution |
| `ABP_PROD_STRIDE`, `ABP_SAVED_PATH_TIME_THIN`, `ABP_MAX_SAVED_PATHS_PER_WINDOW` | Example saving cadence, time resolution, cap |
| `ABP_OUTPUT_DIR`, `ABP_SAVE_CSV` | Output location and CSV export |

The generic traversal target uses `ABP_ROUNDTRIP_AVG_TARGET_FRACTION`;
the fixed-D runner historically also uses
`ABP_ROUNDTRIP_TOTAL_TARGET_FRACTION`. Read resolved settings rather than
inferring a stopping rule from names.

## 10. Longer movies and complete production paths

`movies.ipynb` reads diagnostic snapshots. Set source directories and stage
selection to inspect learning or production. Chain IDs, iteration/local
step data, and physical-time grids distinguish the animation axes. Frames
can repeat after rejections: this is actual chain behavior. Snapshots are
biased examples, not an unbiased trajectory sample.

In its configuration cell, set `MOVIE_RUN_DIR` or an exact `MOVIE_CSV`,
`PHASE="muca"` or `"production"`, and `CHAIN_ID`. `MAX_MARKOV_FRAMES=None`
keeps all saved MCMC frames; `MARKOV_FRAME_THIN`, `PHYSICAL_FRAME_THIN`,
and `FPS` set playback. `N_OVERLAY_PATHS` sets the overview count. Use
`X_LIM=None` and `Y_LIM=None` to display the complete saved spatial extent.
`SAVE_FORMAT="mp4"` or `"gif"` selects the rendering format.

Production controls: `ABP_SAVE_MOVIE_PATHS`, `ABP_MOVIE_CHAIN_ID`,
`ABP_MOVIE_STRIDE`, `ABP_MOVIE_MAX_TRAJECTORIES`,
`ABP_MOVIE_PATH_TIME_THIN`, and endpoint filter
`ABP_MOVIE_ENDPOINT_X_MIN/MAX`. Learning has separate controls
`ABP_SAVE_MUCA_MOVIE_PATHS`, `ABP_MUCA_MOVIE_STRIDE`, and
`ABP_MUCA_MOVIE_MAX_TRAJECTORIES`; it shares the selected chain, stored
physical-time thinning, and endpoint filter. For landscape exploration,
retain **all endpoints** and filter later in plotting.

```bash
ABP_SAVE_MUCA_MOVIE_PATHS=true ABP_MUCA_MOVIE_MAX_TRAJECTORIES=10000 \
ABP_SAVE_MOVIE_PATHS=true ABP_MOVIE_MAX_TRAJECTORIES=10000 \
ABP_MOVIE_PATH_TIME_THIN=1 \
sbatch run/run_abp_movie_paths_array_nariman.sh
```

Caps bound memory/disk use. Increasing physical `T` changes the ensemble
and workload; changing snapshot cadence/caps changes diagnostics only.
The effective snapshot cadence is the larger of the requested stride and
`ceil(planned_sampling_steps / cap)`, so a small budget spans the planned
run rather than filling immediately. The final eligible state is retained,
replacing the last snapshot if the budget is full. MUCA's plan uses its
maximum iteration count; early stopping or endpoint filters can therefore
leave fewer frames. Snapshots skip thermalization, while MUCA `mcmc_step`
includes its cumulative thermalization moves. Production `mcmc_step` starts
after thermalization. Saved `sampling_step` identifies the sampling axis.
Longer movies require more sampled frames, fewer discarded frames, or more
physical-time points per path. Lowering FPS alone adds no sampled paths.
Check notebook frame limits before assuming a movie includes the run's end.
A notebook frame budget of one displays only the final saved state; budgets
of two or more retain both ends of the selected sequence.

The notebooks' full-path panels show x-y, x(t), y(t), and θ(t) for multiple
saved production trajectories. Set `ABP_SAVED_PATH_TIME_THIN=1` for every
integration point in examples, or `ABP_MOVIE_PATH_TIME_THIN=1` for movie
snapshots. Older thinned files cannot recover omitted points. The histogram
spatial filter does not crop saved examples. Capped examples can emphasize
early states or earlier merged chains; use histograms for probabilities.
New snapshots always retain the first and last physical-time points even
if the thinning stride does not divide `N−1`, and store explicit
`physical_times`; use those rather than assuming the final interval has
the usual thinned spacing. Saved examples include both wrapped `thetas`
and `thetas_unwrapped`, plus chain and MCMC-step labels. Set
`MAX_EXAMPLE_PATHS` in the production error notebook to change the number
of displayed examples.

MP4 export needs FFmpeg; GIF uses Pillow and can be large. Rendering changes
no simulation. MUCA-learning snapshots need a new run; legacy production
movies remain usable for production inspection.

## 11. Outputs and code map

Each case writes `abp_endpoint_conditioned_D<tag>.jld2`, with CSVs below
`data/abp_endpoint_conditioned_D<tag>/` inside the output root. Array
collection adds speed tags where needed to prevent filename collisions.

| JLD2 group | Contents |
| --- | --- |
| `metadata/` | Physical, numerical, measurement settings and timings |
| `muca/` | Iteration histograms/logweights, flatness/traversal diagnostics |
| `histograms/` | Edges, endpoint-x masses, weight diagnostics |
| `conditional_windows/<key>/` | Endpoint-conditioned observables |
| `path_endpoint_conditions/<key>/` | Path occupation; x-theta for `x_gt_0p5` |
| `active_work/` | Pooled crossing-y work moments and exclusion counters |
| `chain_statistics/chains/<id>/` | Chain masses/moments for jackknife errors |
| `roundtrips/` | Cumulative production traversal traces |
| `saved_trajectories/<key>/` | Capped production examples |
| `movie_trajectories/` | Production snapshots |
| `muca_movie_trajectories/` | Optional learning snapshots |

CSV names include `endpoint_x_histogram.csv`,
`endpoint_x_gt_0p5_filtered_path_x_theta_long.csv`, and
`movie_trajectories_long.csv`; learning movies use
`muca_movie_trajectories_long.csv`. Read column names: raw masses are not PDFs.
The error notebook needs JLD2 chain statistics, absent from ordinary pooled
CSVs.

| Source | Responsibility |
| --- | --- |
| `model.jl` | Model, potential/force, integration, work/crossing observables |
| `moves.jl` | Proposals, reflection, accept/revert |
| `config.jl`, `cli.jl` | Defaults, validation, environment/CLI |
| `muca.jl`, `muca_utils.jl` | Learning, extension, convergence diagnostics |
| `production.jl` | Frozen-bias chains and observation schedule |
| `production_accumulators.jl` | Occupation/work statistics and snapshot copies |
| `histograms.jl` | Binning, normalization, unbiasing |
| `io.jl` | JLD2/CSV export |
| `driver.jl`, `runtime.jl` | Sweep orchestration and runtime reporting |

The recursive update, flatness, traversal counting, and visit hooks depend
on [MonteCarloX source](https://github.com/zierenberg/MonteCarloX.jl).
The audit above used locally installed source rather than assuming another
release's documentation describes these runs.
