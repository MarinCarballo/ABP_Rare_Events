#!/usr/bin/env julia

# One fixed-noise, one-speed task:
#   MUCA -> freeze learned weights -> production -> JLD2/CSV output.

import Pkg

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)

using ABPNoiseSweep

env_int(name::AbstractString, default::Integer) =
    parse(Int, get(ENV, name, string(default)))

env_float(name::AbstractString, default::Real) =
    parse(Float64, get(ENV, name, string(default)))

function env_bool(name::AbstractString, default::Bool)
    value = lowercase(strip(get(ENV, name, string(default))))
    return value in ("1", "true", "yes", "y", "on")
end

number_tag(value::Real) = replace(string(Float64(value)), "." => "p", "-" => "m")

# -----------------------------------------------------------------------------
# Fixed physical parameter and array-selected speed
# -----------------------------------------------------------------------------

const D = 0.025

haskey(ENV, "ABP_V") || error(
    "ABP_V is not set. The Slurm array script must select one propulsion speed."
)
const V = parse(Float64, ENV["ABP_V"])

const OUTPUT_DIR = get(
    ENV,
    "ABP_OUTPUT_DIR",
    joinpath("data", "fixed_D0p025_speed_landscape", "v_$(number_tag(V))"),
)

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------

const N_MUCA_ITER = env_int("ABP_N_ITER", 200)
const MUCA_MOVES_LAST_ITER =
    env_int("ABP_N_ITER_STEPS_PER_ITER", 800_000_000)
const N_THERM_MUCA =
    env_int("ABP_N_THERM_MUCA", 1_000_000)

const N_PRODUCTION_MOVES =
    env_int("ABP_N_PROD_OBS_TOTAL", 50_000_000_000)
const N_THERM_PRODUCTION =
    env_int("ABP_N_THERM_PROD", 1_000_000)
const N_PRODUCTION_CHAINS = min(
    env_int("ABP_N_PROD_CHAINS", Threads.nthreads()),
    Threads.nthreads(),
)

const PROD_STRIDE =
    env_int("ABP_PROD_STRIDE", 10_000)
const ROUNDTRIP_STRIDE =
    env_int("ABP_ROUNDTRIP_STRIDE", 10_000)

const PATH_OBSERVATION_STRIDE =
    env_int("ABP_PATH_OBSERVATION_STRIDE", 1_000)
const PATH_TIME_STRIDE =
    env_int("ABP_PATH_TIME_STRIDE", 1)
const PATH_FILTER_X_MIN =
    env_float("ABP_PATH_FILTER_X_MIN", -1.0)

const ROUNDTRIP_TOTAL_TARGET_FRACTION =
    env_float("ABP_ROUNDTRIP_TOTAL_TARGET_FRACTION", 0.5)
const ROUNDTRIP_CONVERGENCE_HITS =
    env_int("ABP_ROUNDTRIP_CONVERGENCE_HITS", 5)
const ROUNDTRIP_TARGET =
    env_int("ABP_ROUNDTRIP_TARGET", 50)
    
cfg = ABPNoiseSweepConfig(
    seed = 42,

    # Physical model: D is fixed; only V changes between array tasks.
    trajectory_T = 40.0,
    dt = 1e-2,
    v = V,
    D_values = Float64[D],
    potential_active = true,

    # MUCA on x(T).  No D-dependent scaling because D is fixed.
    bias_min = -1.8,
    bias_max = 1.5,
    dbias = 0.05,
    xT_min = -1.0,
    xT_max = 1.0,
    xT_extension_margin = 0.2,
    n_iter = N_MUCA_ITER,
    n_iter_steps_per_iter = MUCA_MOVES_LAST_ITER,
    n_therm_muca = N_THERM_MUCA,
    D_scaling_reference = D,
    scale_n_iter_with_D = false,
    block_dxi = 0.05,
    local_dxi = 0.8,
    move_weights = Float64[0.0, 0.05, 0.55, 0.40],

    # Complete the modest MUCA schedule unless transport is exceptionally good.
    roundtrip_target = 50,
    roundtrip_avg_target_fraction = 1.0,
    roundtrip_convergence_hits = 3,

    # Production begins immediately after MUCA in this same Julia process.
    production_parallel = true,
    n_prod_chains = N_PRODUCTION_CHAINS,
    n_therm_prod = N_THERM_PRODUCTION,
    n_prod_obs_total = N_PRODUCTION_MOVES,
    prod_stride = 1_000,
    roundtrip_stride = 1_000,

    # Whole-path landscape conditioned on x(T)>0.5.
    path_observation_stride = PATH_OBSERVATION_STRIDE,
    path_time_stride = PATH_TIME_STRIDE,
    path_filter_x_min = PATH_FILTER_X_MIN,
    saved_path_time_thin = 5,
    max_saved_paths_per_window = 120,

    # Histogram resolution.
    y_abs = 2.0,
    n_y_bins = 241,
    n_y_int_bins = 241,
    path_x_min = -1.5,
    path_x_max = 1.5,
    n_path_x_bins = 241,

    output_dir = OUTPUT_DIR,
    save_csv = env_bool("ABP_SAVE_CSV", true),
    save_plots = false,
    show_plots = false,
)

abp_validate_config!(cfg)

approx_muca_moves = round(
    Int,
    MUCA_MOVES_LAST_ITER * (N_MUCA_ITER + 1) / 2,
)

println("============================================================")
println("Fixed-noise ABP speed landscape: MUCA -> production")
println("D = Dt = Dr = ", D, " | v = ", V)
println("Julia threads = ", Threads.nthreads())
println("MUCA iterations = ", N_MUCA_ITER)
println("Approximate total MUCA sampling moves = ", approx_muca_moves)
println("Production moves = ", N_PRODUCTION_MOVES)
println("Production chains = ", N_PRODUCTION_CHAINS)
println("Path-observation stride = ", PATH_OBSERVATION_STRIDE)
println("Output = ", OUTPUT_DIR)
println("============================================================")

# D_values contains exactly one value.  The package therefore runs one MUCA
# stage and then production immediately, using the newly learned weights.
result_files, data_files = run_abp_noise_sweep(cfg)

println()
println("SUCCESS: MUCA and production completed for v = ", V)
println("Main result: ", only(result_files))
println("CSV files written: ", length(data_files))
