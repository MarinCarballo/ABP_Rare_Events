# Changes and audit report

The potential, forces, diffusion/noise conventions, observables, and default
Euler integration and random-walk proposals are preserved. The requested speed
grid and optional numerical/sampling methods are explicit choices.

| Request | Completed work |
| --- | --- |
| Documentation | Replaced the outdated README and added [equations and usage](docs/physics_and_usage.md): model, integration, path-space MCMC, MUCA recursion, reweighting, histogram normalization, errors, output formats, and cluster/notebook instructions. |
| Longer movies | Added optional snapshots during MUCA learning, with iteration and actual attempted-move labels. Bounded capture spans the planned run and retains the final eligible state. Reworked `movies.ipynb` for MUCA/production evolution, physical-time animations, MP4/GIF export, and configurable frame limits. |
| Active work and “flat” | Verified the existing full-resolution midpoint work calculation and first-crossing interpolation. Distinguished the notebook’s flat **route** cutoff from MUCA histogram **flatness**; see findings below. |
| Full production trajectories | Added x-y and x(t), y(t), theta(t) panels for multiple saved paths. Storage can keep every integration point; snapshots retain the physical endpoint and explicit time coordinates. |
| Crank–Nicolson option | Added two separate options: `ABP_INTEGRATOR=crank_nicolson` for physical-time integration, and `ABP_NOISE_PROPOSAL=pcn` for Gaussian-prior-preserving noise proposals. Euler/random-walk remain defaults. |
| Cleanup and allocations | Shared snapshot/export helpers, streamed movie CSV rows, reused noise buffers and path-window lists, and replaced suffix slices with in-place copies. Seeded legacy transitions and random draws are regression checked. |
| Speeds below 0.32 | Main speed/work-theta arrays now cover **0.280:0.005:0.390** (23 tasks); the low-speed array covers **0.280:0.005:0.325**. The supplied critical speed was used as a reference, not recalculated. |
| Theta histograms | Verified modulo-2pi binning, path filtering, endpoint conditioning, whole-path reweighting, and density normalization. No histogram definitions changed. |

Important findings:

- **Work:** matches `W_a = v sum e(theta_mid) dot delta_r`, using unwrapped
  angles, every integration step, and unit mobility. Production summaries
  condition on `x(T)>0.5` and a valid first left-to-right crossing of `x=0`.
  Normalization remains `W_a/(v^2*T)`; finite-step conventions are documented.
- **Flat route:** the notebook classifies routes using the first-crossing-y
  histogram bin centers. Removed a hidden `M_CUT=1` assignment that overrode
  the configuration cell’s `M_CUT=0.5`. Regenerated route plots now honor the
  chosen configuration and can differ from previously displayed plots.
- **MUCA flatness:** max/mean and mean/min use **positive bins only**.
  Empty bins are ignored; one occupied bin can report 1. The flatness
  threshold does not stop learning. Stopping uses consecutive-iteration
  boundary-traversal targets; “roundtrips” count one-way traversals.
- **Existing sampling caveat:** prior/domain rejections skip the dependency’s
  MUCA visit hook, and some attempted moves are counted twice in its acceptance
  denominator. These affect interpretation and potentially bias learning.
  They are documented and regression captured; the learning policy is
  preserved. This review does not certify MUCA convergence.
- **Theta/errors:** theta is an occupation distribution
  `P(x(t),theta(t) mod 2pi | x(T)>0.5, x(t)>filter)`, not endpoint theta.
  Finite ranges further condition normalized densities. Weight-only ESS ignores
  autocorrelation; chain-delete jackknife errors omit MUCA-learning and
  time-discretization uncertainty.

Warmed measurements at `T=40` reduced local/block proposal allocations from
approximately **25.6/36.7 kB per move to 0 bytes** in the tested harness.
This is not a whole-program allocation or runtime guarantee.

Fresh installation exposed an incompatible upstream MonteCarloX 0.3 API.
`Project.toml` now pins the compatible 0.2 source revision. Also corrected the
fixed-speed wrapper’s missing output-directory assignment and a stray Markdown
fence that prevented `run_combined.sh` from parsing.

Validation: **1,977 Julia checks passed**, including seeded transition equality,
CN/pCN checks, work/theta calculations, snapshot independence, and threaded tiny
MUCA-to-production exports. Notebook syntax/statistics/rendering and cluster
shell syntax were also checked. No long run or cluster job was submitted.
New MUCA movies require newly recorded snapshots; old thinned trajectories
cannot recover missing integration points. CN solves the same SDE with a
different finite-step approximation and needs time-step convergence checks.

Added a mode-based work comparison with the user-confirmed displacement
`FLAT_THEORY_DISPLACEMENT=2.1`: flat reference `2.1v` and SB references
`2.1v*cos(theta_peak)` from both resolved modal branches of
`SB_THEORY_THETA_SLICE="crossing"`. Missing modes remain gaps. Raw and
normalized panels now share these settings; corrected the reference-only
`v**2.1*T` denominator to `v**2*T`. These are constant-angle comparison
curves based on occupation modes, not a proof of optimal trajectories or
m-specific work predictions. See [the definitions and assumptions](docs/active_work_components.md).

Added separate fits to the selected crossing-y slices: `near_m1` fits both
displacements in `W=v*(delta_x*cos(theta_peak)+delta_y*sin(theta_peak))`;
`near_m0` fits `W=v*delta_x` with no intercept. Both provide raw and
`W/(v^2*T)` comparison figures and parameter/prediction CSVs. Parameter
errors condition on the supplied modal angles and omit their uncertainty.
These fits are postprocessing; the simulation physics and measured work are
unchanged.

Start with the [README](README.md); full controls and limitations are in the
[detailed documentation](docs/physics_and_usage.md).
