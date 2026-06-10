# CLI and environment-variable interface.

function abp_parse_float_vector(s::AbstractString)
    vals = [strip(x) for x in split(s, ',') if !isempty(strip(x))]
    return Float64[parse(Float64, x) for x in vals]
end

function abp_env_bool(name::AbstractString, default::Bool)
    s = lowercase(strip(get(ENV, name, string(default))))
    return s in ("1", "true", "yes", "y", "on")
end

function abp_apply_env_overrides!(cfg::ABPNoiseSweepConfig)
    haskey(ENV, "ABP_D_VALUES") && (cfg.D_values = abp_parse_float_vector(ENV["ABP_D_VALUES"]))
    haskey(ENV, "ABP_MOVE_WEIGHTS") && (cfg.move_weights = abp_parse_float_vector(ENV["ABP_MOVE_WEIGHTS"]))
    haskey(ENV, "ABP_OUTPUT_DIR") && (cfg.output_dir = ENV["ABP_OUTPUT_DIR"])

    haskey(ENV, "ABP_N_ITER") && (cfg.n_iter = parse(Int, ENV["ABP_N_ITER"]))
    haskey(ENV, "ABP_N_ITER_STEPS_PER_ITER") && (cfg.n_iter_steps_per_iter = parse(Int, ENV["ABP_N_ITER_STEPS_PER_ITER"]))
    haskey(ENV, "ABP_N_THERM_MUCA") && (cfg.n_therm_muca = parse(Int, ENV["ABP_N_THERM_MUCA"]))
    haskey(ENV, "ABP_D_SCALING_REFERENCE") && (cfg.D_scaling_reference = parse(Float64, ENV["ABP_D_SCALING_REFERENCE"]))
    haskey(ENV, "ABP_SCALE_N_ITER_WITH_D") && (cfg.scale_n_iter_with_D = abp_env_bool("ABP_SCALE_N_ITER_WITH_D", cfg.scale_n_iter_with_D))
    haskey(ENV, "ABP_BLOCK_DXI") && (cfg.block_dxi = parse(Float64, ENV["ABP_BLOCK_DXI"]))
    haskey(ENV, "ABP_LOCAL_DXI") && (cfg.local_dxi = parse(Float64, ENV["ABP_LOCAL_DXI"]))

    haskey(ENV, "ABP_N_PROD_OBS_TOTAL") && (cfg.n_prod_obs_total = parse(Int, ENV["ABP_N_PROD_OBS_TOTAL"]))
    haskey(ENV, "ABP_N_PROD_CHAINS") && (cfg.n_prod_chains = parse(Int, ENV["ABP_N_PROD_CHAINS"]))
    haskey(ENV, "ABP_N_THERM_PROD") && (cfg.n_therm_prod = parse(Int, ENV["ABP_N_THERM_PROD"]))
    haskey(ENV, "ABP_PROD_STRIDE") && (cfg.prod_stride = parse(Int, ENV["ABP_PROD_STRIDE"]))
    haskey(ENV, "ABP_ROUNDTRIP_STRIDE") && (cfg.roundtrip_stride = parse(Int, ENV["ABP_ROUNDTRIP_STRIDE"]))

    haskey(ENV, "ABP_ROUNDTRIP_TARGET") && (cfg.roundtrip_target = parse(Int, ENV["ABP_ROUNDTRIP_TARGET"]))
    haskey(ENV, "ABP_ROUNDTRIP_CONVERGENCE_WINDOW") && (cfg.roundtrip_convergence_window = parse(Int, ENV["ABP_ROUNDTRIP_CONVERGENCE_WINDOW"]))
    haskey(ENV, "ABP_ROUNDTRIP_CONVERGENCE_RTOL") && (cfg.roundtrip_convergence_rtol = parse(Float64, ENV["ABP_ROUNDTRIP_CONVERGENCE_RTOL"]))

    haskey(ENV, "ABP_SAVE_CSV") && (cfg.save_csv = abp_env_bool("ABP_SAVE_CSV", cfg.save_csv))
    haskey(ENV, "ABP_SAVE_PLOTS") && (cfg.save_plots = abp_env_bool("ABP_SAVE_PLOTS", cfg.save_plots))
    haskey(ENV, "ABP_SHOW_PLOTS") && (cfg.show_plots = abp_env_bool("ABP_SHOW_PLOTS", cfg.show_plots))

    return cfg
end

function abp_config_from_args(args=ARGS)
    if "--smoke" in args
        cfg = ABPNoiseSweepConfig(
            D_values=Float64[0.01, 0.1],
            n_iter=3,
            n_iter_steps_per_iter=300_000,
            n_therm_muca=1_000,
            n_prod_obs_total=50_000,
            n_therm_prod=5_000,
            prod_stride=1_000,
            roundtrip_stride=1_000,
            output_dir="abp_noise_sweep_smoke_test",
            save_csv=true,
            save_plots=false,
            show_plots=false,
        )
    else
        cfg = ABPNoiseSweepConfig()
    end

    "--no-plots" in args && (cfg.save_plots = false)
    "--no-csv" in args && (cfg.save_csv = false)
    "--show-plots" in args && (cfg.show_plots = true)

    return abp_apply_env_overrides!(cfg)
end

function abp_print_iteration_schedule(cfg::ABPNoiseSweepConfig)
    println("Derived MUCA iteration schedule:")
    println("  Formula: n_iter_effective(D) = round(n_iter * sqrt(D_scaling_reference / D))")
    println("  n_iter = ", cfg.n_iter)
    println("  D_scaling_reference = ", cfg.D_scaling_reference)
    println("  scale_n_iter_with_D = ", cfg.scale_n_iter_with_D)
    for row in abp_iteration_schedule_rows(cfg)
        println("  D=", row.D,
                " | factor=", round(row.n_iter_factor; sigdigits=5),
                " | effective n_iter=", row.n_iter_effective,
                " | final-iteration sampling moves≈", row.final_iteration_sampling_moves,
                " | total sampling moves over MUCA ramp≈", round(Int, row.approx_total_sampling_moves))
    end
    return nothing
end

function abp_print_config(cfg::ABPNoiseSweepConfig)
    println("Configuration:")
    println("  D_values = ", cfg.D_values)
    println("  move_weights [reflection, theta0, block, local] = ", cfg.move_weights)
    println("  trajectory_T = ", cfg.trajectory_T, " | dt = ", cfg.dt, " | v = ", cfg.v)
    println("  potential_active = ", cfg.potential_active)
    println("  n_iter = ", cfg.n_iter)
    println("  n_iter_steps_per_iter = ", cfg.n_iter_steps_per_iter,
            " final-iteration sampling moves across chains")
    println("  n_therm_muca = ", cfg.n_therm_muca)
    println("  D_scaling_reference = ", cfg.D_scaling_reference)
    println("  scale_n_iter_with_D = ", cfg.scale_n_iter_with_D)
    println("  block_dxi = ", cfg.block_dxi)
    println("  local_dxi = ", cfg.local_dxi)
    println("  n_prod_obs_total = ", cfg.n_prod_obs_total)
    println("  n_prod_chains = ", cfg.n_prod_chains)
    println("  roundtrip_target = ", cfg.roundtrip_target)
    println("  roundtrip_convergence_window = ", cfg.roundtrip_convergence_window)
    println("  roundtrip_convergence_rtol = ", cfg.roundtrip_convergence_rtol)
    println("  output_dir = ", cfg.output_dir)
    println("  save_csv = ", cfg.save_csv)
    println("  save_plots = ", cfg.save_plots, " (ignored by data-only script)")
    abp_print_iteration_schedule(cfg)
    return nothing
end

function main(args=ARGS)
    abp_print_runtime_header()
    cfg = abp_config_from_args(args)
    abp_validate_config!(cfg)
    abp_print_config(cfg)

    if "--dry-run" in args
        println("Dry run requested; no simulation was started.")
        return nothing
    end

    run_abp_noise_sweep(cfg)
    println("Finished at: ", Dates.format(now(), dateformat"yyyy-mm-dd HH:MM:SS"))
    return nothing
end
