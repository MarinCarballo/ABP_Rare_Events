# User-editable configuration and D-dependent scheduling helpers.

Base.@kwdef mutable struct ABPNoiseSweepConfig
    seed::Int = 42

    # Physical ABP trajectory integration time. 
    trajectory_T::Float64 = 20.0
    dt::Float64 = 1e-2
    v::Float64 = 0.36
    x0_vec::Vector{Float64} = Float64[-1.0, 0.0]
    potential_active::Bool = true

    # Noise sweep. D is used as Dt = Dr = D.
    D_values::Vector{Float64} = Float64[0.1, 0.01, 0.005]

    # Endpoint-x MUCA bias.
    bias_min::Float64 = -1.8
    bias_max::Float64 =  1.5
    dbias::Float64 = 0.025

    # Diagnostic/extension anchors in x(T).
    xT_min::Float64 = -1.0
    xT_max::Float64 =  1.0
    xT_extension_margin::Float64 = 0.2

    # MUCA iteration schedule.
    # If scale_n_iter_with_D=true, the number of recursive MUCA iterations is
    # n_iter_effective(D) = round(n_iter * sqrt(D_scaling_reference / D)).
    # The ramp of sampling moves is stretched over the effective number of
    # iterations, so the final iteration still uses approximately
    # n_iter_steps_per_iter total sampling moves across chains.
    n_iter::Int = 90
    n_iter_steps_per_iter::Int = 120_000_000
    n_therm_muca::Int = 100_000
    D_scaling_reference::Float64 = 0.01
    scale_n_iter_with_D::Bool = true

    # Path-space proposal amplitudes in standardized noise variables ξ.
    # These are fixed across D by default. Reflection is controlled only through
    # move_weights below.
    block_dxi::Float64 = 0.05
    local_dxi::Float64 = 0.8

    # Move weights in this order:
    # [reflection_update, theta0_update, block_noise_update, local_noise_update]
    # Default has no reflection. To include reflection, increase the first weight
    # and reduce another weight so the vector remains nonnegative.
    move_weights::Vector{Float64} = Float64[0.0, 0.05, 0.55, 0.40]

    flatness_threshold::Float64 = 2.0

    # Production is split over independent chains when production_parallel=true.
    # Total production samples across chains = n_prod_obs_total.
    production_parallel::Bool = true
    n_prod_chains::Int = Threads.nthreads()
    n_therm_prod::Int = 100_000
    n_prod_obs_total::Int = 60_000_000
    prod_stride::Int = 10_000
    roundtrip_stride::Int = 10_000

    # Roundtrip convergence control.
    # The MUCA learning loop stops early when the average number of completed
    # roundtrips per chain reaches roundtrip_avg_target_fraction * n_chains for
    # roundtrip_convergence_hits consecutive iterations.
    roundtrip_target::Int = 100
    roundtrip_avg_target_fraction::Float64 = 0.5
    roundtrip_convergence_hits::Int = 3

    # Whole-trajectory counting. path_time_stride=1 means every saved integration
    # point in the trajectory is counted when x(T)>0.
    path_time_stride::Int = 1

    # Storage thinning only; does not affect histograms.
    saved_path_time_thin::Int = 5
    max_saved_paths_per_window::Int = 300

    # Histogram ranges for y and path heatmaps.
    y_abs::Float64 = 6.0
    n_y_bins::Int = 241
    n_y_int_bins::Int = 241
    path_x_min::Float64 = -1.5
    path_x_max::Float64 =  1.5
    n_path_x_bins::Int = 241

    # Output. This project writes JLD2 plus CSV data exports; plotting is separate.
    output_dir::String = "abp_noise_sweep_endpoint_conditioned"
    save_csv::Bool = true
    save_plots::Bool = false
    show_plots::Bool = false
end

# -----------------------------
# D-dependent scheduling helpers
# -----------------------------

function abp_D_scaling_factor(cfg::ABPNoiseSweepConfig, D::Real)
    @assert cfg.D_scaling_reference > 0.0 "D_scaling_reference must be positive."
    @assert D > 0.0 "D must be positive."
    return sqrt(cfg.D_scaling_reference / Float64(D))
end

function abp_n_iter_factor_for_D(cfg::ABPNoiseSweepConfig, D::Real)
    return cfg.scale_n_iter_with_D ? abp_D_scaling_factor(cfg, D) : 1.0
end

function abp_n_iter_for_D(cfg::ABPNoiseSweepConfig, D::Real)
    return max(1, round(Int, cfg.n_iter * abp_n_iter_factor_for_D(cfg, D)))
end

function abp_iteration_schedule_rows(cfg::ABPNoiseSweepConfig)
    return [(
        D = D,
        n_iter_factor = abp_n_iter_factor_for_D(cfg, D),
        n_iter_effective = abp_n_iter_for_D(cfg, D),
        final_iteration_sampling_moves = cfg.n_iter_steps_per_iter,
        approx_total_sampling_moves = cfg.n_iter_steps_per_iter * (abp_n_iter_for_D(cfg, D) + 1) / 2,
    ) for D in cfg.D_values]
end

function abp_validate_config!(cfg::ABPNoiseSweepConfig)
    isempty(cfg.D_values) && error("D_values must not be empty.")
    any(D -> D <= 0.0, cfg.D_values) && error("All D values must be positive.")
    cfg.dt <= 0.0 && error("dt must be positive.")
    cfg.trajectory_T <= 0.0 && error("trajectory_T must be positive.")
    cfg.n_iter < 1 && error("n_iter must be at least 1.")
    cfg.n_iter_steps_per_iter < 1 && error("n_iter_steps_per_iter must be positive.")
    cfg.n_therm_muca < 0 && error("n_therm_muca must be non-negative.")
    cfg.D_scaling_reference <= 0.0 && error("D_scaling_reference must be positive.")
    cfg.block_dxi <= 0.0 && error("block_dxi must be positive.")
    cfg.local_dxi <= 0.0 && error("local_dxi must be positive.")
    length(cfg.move_weights) == 4 || error("move_weights must contain exactly four entries: [reflection, theta0, block, local].")
    any(w -> w < 0.0, cfg.move_weights) && error("move_weights must be nonnegative.")
    sum(cfg.move_weights) > 0.0 || error("At least one move weight must be positive.")
    cfg.n_prod_obs_total < 1 && error("n_prod_obs_total must be at least 1.")
    cfg.n_prod_chains < 1 && error("n_prod_chains must be at least 1.")
    cfg.n_therm_prod < 0 && error("n_therm_prod must be non-negative.")
    cfg.prod_stride < 1 && error("prod_stride must be at least 1.")
    cfg.roundtrip_stride < 1 && error("roundtrip_stride must be at least 1.")
    cfg.path_time_stride < 1 && error("path_time_stride must be at least 1.")
    cfg.saved_path_time_thin < 1 && error("saved_path_time_thin must be at least 1.")
    cfg.max_saved_paths_per_window < 0 && error("max_saved_paths_per_window must be non-negative.")
    cfg.bias_min >= cfg.bias_max && error("bias_min must be smaller than bias_max.")
    cfg.dbias <= 0.0 && error("dbias must be positive.")
    cfg.xT_min >= cfg.xT_max && error("xT_min must be smaller than xT_max.")
    cfg.xT_extension_margin < 0.0 && error("xT_extension_margin must be non-negative.")
    cfg.bias_min > cfg.xT_min - cfg.xT_extension_margin && error("bias_min must cover xT_min - xT_extension_margin.")
    cfg.bias_max < cfg.xT_max + cfg.xT_extension_margin && error("bias_max must cover xT_max + xT_extension_margin.")
    cfg.y_abs <= 0.0 && error("y_abs must be positive.")
    cfg.n_y_bins < 1 && error("n_y_bins must be positive.")
    cfg.n_y_int_bins < 1 && error("n_y_int_bins must be positive.")
    cfg.path_x_min >= cfg.path_x_max && error("path_x_min must be smaller than path_x_max.")
    cfg.n_path_x_bins < 1 && error("n_path_x_bins must be positive.")
    return cfg
end
