# Production sampling using the learned MUCA bias.

function abp_matches_endpoint_spec(xT::Real, spec)
    lower_ok = spec.lower_strict ? xT > spec.lo : xT >= spec.lo
    upper_ok = xT < spec.hi
    return lower_ok && upper_ok
end

function abp_run_production_one_case(cfg::ABPNoiseSweepConfig, D::Real, muca)
    obs = muca.obs
    bins_bias = muca.bins_bias
    edges_bias = muca.edges_bias
    learned_logw_values = vec(Float64.(logweight(muca.alg).values))
    finite_logw = isfinite.(learned_logw_values)
    @assert any(finite_logw) "All learned MUCA logweights are non-finite."
    logw_shift = minimum(learned_logw_values[finite_logw])

    edges_x_T = copy(edges_bias)
    edges_y_T = collect(range(-cfg.y_abs, cfg.y_abs, length=cfg.n_y_bins + 1))
    edges_y_mean = copy(edges_y_T)
    edges_y_int = collect(range(-cfg.y_abs * cfg.trajectory_T, cfg.y_abs * cfg.trajectory_T, length=cfg.n_y_int_bins + 1))
    edges_path_x = collect(range(cfg.path_x_min, cfg.path_x_max, length=cfg.n_path_x_bins + 1))
    edges_path_y = copy(edges_y_T)

    # Endpoint-conditioned one-dimensional observables and saved examples.
    endpoint_window_specs = [
        (key=:x_m1_0,  lo=-1.0, hi=0.0, lower_strict=false, label="-1 ≤ x(T) < 0"),
        (key=:x_gt0,    lo=0.0, hi=Inf, lower_strict=true,  label="x(T) > 0"),
        (key=:x_gt_0p5, lo=0.5, hi=Inf, lower_strict=true,  label="x(T) > 0.5"),
        (key=:all,     lo=-Inf, hi=Inf, lower_strict=false, label="all endpoints"),
    ]
    n_windows = length(endpoint_window_specs)
    window_keys = [string(w.key) for w in endpoint_window_specs]

    # Whole-path occupation histograms. Every kept trajectory point satisfies
    # x(t) > path_x_min; endpoint selection is given by these three windows.
    path_x_min = cfg.path_filter_x_min
    path_endpoint_specs = [
        (key=:all,      lo=-Inf, hi=Inf, lower_strict=false, label="all endpoints"),
        (key=:x_gt0,     lo=0.0, hi=Inf, lower_strict=true,  label="x(T) > 0"),
        (key=:x_gt_0p5,  lo=0.5, hi=Inf, lower_strict=true,  label="x(T) > 0.5"),
    ]
    n_path_windows = length(path_endpoint_specs)
    path_window_keys = [string(w.key) for w in path_endpoint_specs]

    n_prod_chains = cfg.production_parallel ? max(1, cfg.n_prod_chains) : 1
    n_prod_chains = min(n_prod_chains, cfg.n_prod_obs_total)
    n_steps_chain = [cfg.n_prod_obs_total ÷ n_prod_chains for _ in 1:n_prod_chains]
    for i in 1:(cfg.n_prod_obs_total % n_prod_chains)
        n_steps_chain[i] += 1
    end

    accs = [
        abp_make_prod_accumulator(edges_x_T, edges_y_T, edges_y_mean, edges_y_int, edges_path_x, edges_path_y, n_windows, n_path_windows)
        for _ in 1:n_prod_chains
    ]

    moves_prod, move_weights_prod = make_abp_moves(
        obs,
        bins_bias,
        cfg.potential_active;
        move_weights=cfg.move_weights,
        block_dxi=muca.block_dxi,
        local_dxi=muca.local_dxi,
    )

    production_rt_min = cfg.xT_min
    production_rt_max = cfg.xT_max
    max_saved_paths_per_window_chain = max(1, ceil(Int, cfg.max_saved_paths_per_window / n_prod_chains))

    println()
        println("Production case: D=", D,
            " | chains=", n_prod_chains,
            " | total samples=", cfg.n_prod_obs_total,
            " | proposal Δξ block=", round(muca.block_dxi; sigdigits=4),
            " | proposal Δξ local=", round(muca.local_dxi; sigdigits=4),
            " | whole-path endpoint windows=", join(path_window_keys, ","),
            " | path filter x(t)>", path_x_min)
    println("Whole-path production observation stride = ", cfg.path_observation_stride)
    println("Whole-path histogram time stride = ", cfg.path_time_stride,
            cfg.path_time_stride == 1 ? " (all integration points counted)" : " (subsampled time points)")
    if cfg.save_movie_paths
        println("Movie path snapshots: chain=", cfg.movie_chain_id,
                " | stride=", cfg.movie_stride,
                " | time thin=", cfg.movie_path_time_thin,
                " | max trajectories=", cfg.movie_max_trajectories,
                " | endpoint window=[", cfg.movie_endpoint_x_min, ", ", cfg.movie_endpoint_x_max, ")")
    end

    t_prod = @elapsed begin
        if n_prod_chains == 1
            @showprogress 1 "Production D=$(D)..." for chain_id in 1:n_prod_chains
                abp_run_production_chain!(
                    accs[chain_id], cfg, D, muca.alg, muca.abp, obs, bins_bias,
                    edges_bias, learned_logw_values, logw_shift,
                    edges_x_T, edges_y_T, edges_y_mean, edges_y_int, edges_path_x, edges_path_y,
                    endpoint_window_specs, path_endpoint_specs, path_x_min,
                    chain_id, n_steps_chain[chain_id], production_rt_min, production_rt_max,
                    moves_prod, move_weights_prod, max_saved_paths_per_window_chain,
                )
            end
        else
            Threads.@threads for chain_id in 1:n_prod_chains
                abp_run_production_chain!(
                    accs[chain_id], cfg, D, muca.alg, muca.abp, obs, bins_bias,
                    edges_bias, learned_logw_values, logw_shift,
                    edges_x_T, edges_y_T, edges_y_mean, edges_y_int, edges_path_x, edges_path_y,
                    endpoint_window_specs, path_endpoint_specs, path_x_min,
                    chain_id, n_steps_chain[chain_id], production_rt_min, production_rt_max,
                    moves_prod, move_weights_prod, max_saved_paths_per_window_chain,
                )
            end
        end
    end

    acc = accs[1]
    for i in 2:length(accs)
        abp_merge_prod_accumulators!(
            acc, accs[i];
            max_saved_paths_per_window=cfg.max_saved_paths_per_window,
            max_movie_paths=cfg.movie_max_trajectories,
        )
    end

    sum_w_all = acc.sum_w_all[1]
    sum_w2_all = acc.sum_w2_all[1]
    ess_all = sum_w2_all > 0.0 ? sum_w_all^2 / sum_w2_all : NaN

    println("Production complete for D=", D,
            " | elapsed=", round(t_prod; digits=1), "s",
            " | reweighted samples=", acc.n_rew_all[1], "/", cfg.n_prod_obs_total,
            " | overall ESS=", ess_all,
            " | ESS/N=", ess_all / cfg.n_prod_obs_total,
            " | aggregate roundtrips=", acc.rt_final_count[1])

    println("Endpoint-window diagnostics:")
    for (iw, spec) in enumerate(endpoint_window_specs)
        ess_i = acc.sum_w2_by_window[iw] > 0.0 ? acc.sum_w_by_window[iw]^2 / acc.sum_w2_by_window[iw] : NaN
        println("  ", spec.label)
        println("    biased samples in window     = ", acc.n_biased_by_window[iw])
        println("    reweighted samples in window = ", acc.n_rew_by_window[iw])
        println("    conditional ESS              = ", ess_i)
        println("    out of range yT/meanY/intY/xTyT = ",
                acc.n_y_T_out[iw], "/", acc.n_y_mean_out[iw], "/", acc.n_y_int_out[iw], "/", acc.n_xy_T_out[iw])
    end
    println("Whole-path endpoint-window diagnostics (kept points satisfy x(t)>", path_x_min, "):")
    for (iw, spec) in enumerate(path_endpoint_specs)
        println("  ", spec.label)
        println("    selected trajectories biased/unbiased = ",
                acc.n_path_traj_biased[iw], "/", acc.n_path_traj_unbiased[iw])
        println("    out of range path y/xy biased = ",
                acc.n_path_y_out_biased[iw], "/", acc.n_path_xy_out_biased[iw])
        println("    out of range path y/xy unbiased = ",
                acc.n_path_y_out_unbiased[iw], "/", acc.n_path_xy_out_unbiased[iw])
    end
    return (
        acc = acc,
        edges_x_T = edges_x_T,
        edges_y_T = edges_y_T,
        edges_y_mean = edges_y_mean,
        edges_y_int = edges_y_int,
        edges_path_x = edges_path_x,
        edges_path_y = edges_path_y,
        endpoint_window_specs = endpoint_window_specs,
        window_keys = window_keys,
        path_endpoint_specs = path_endpoint_specs,
        path_window_keys = path_window_keys,
        path_x_min = path_x_min,
        learned_logw_values = learned_logw_values,
        logw_shift = logw_shift,
        production_rt_min = production_rt_min,
        production_rt_max = production_rt_max,
        n_prod_chains = n_prod_chains,
        n_steps_chain = n_steps_chain,
        t_prod = t_prod,
    )
end

function abp_run_production_chain!(
    acc, cfg::ABPNoiseSweepConfig, D::Real, alg_template, abp, obs, bins_bias,
    learned_bias_edges, learned_logw_values, logw_shift,
    edges_x_T, edges_y_T, edges_y_mean, edges_y_int, edges_path_x, edges_path_y,
    endpoint_window_specs, path_endpoint_specs, path_x_min::Real,
    chain_id::Int, n_prod_steps::Int, production_rt_min::Real, production_rt_max::Real,
    moves_prod, move_weights_prod, max_saved_paths_per_window_chain::Int,
)
    sys = ABPTrajectory(
        Xoshiro(cfg.seed + 700_000 + 1_000_000 * round(Int, 10_000 * D) + 10_000 * chain_id),
        abp,
        cfg.trajectory_T;
        dt=cfg.dt,
        x0=cfg.x0_vec,
        θ0=:uniform,
        potential_active=cfg.potential_active,
    )

    alg = deepcopy(alg_template)
    try
        Random.seed!(alg.rng, cfg.seed + 710_000 + 1_000_000 * round(Int, 10_000 * D) + 10_000 * chain_id)
    catch err
        @warn "Could not reseed copied production algorithm RNG; continuing with copied RNG state." exception=(err, catch_backtrace())
    end

    for _ in 1:cfg.n_therm_prod
        random_move!(sys, alg, moves_prod, move_weights_prod)
    end
    reset!(alg)

    rt_prod = Roundtrips(production_rt_min, production_rt_max)

    for mcmc_step in 1:n_prod_steps
        random_move!(sys, alg, moves_prod, move_weights_prod)

        bias_value_now = obs(sys)
        xT_now    = endpoint_x(sys)
        yT_now    = endpoint_y(sys)
        ymean_now = mean_y(sys)
        yint_now  = path_y_int(sys)

        w_path = abp_unbias_weight_from_value(
            bias_value_now,
            learned_bias_edges,
            learned_logw_values,
            logw_shift,
        )


        if cfg.save_movie_paths &&
           chain_id == cfg.movie_chain_id &&
           mcmc_step % cfg.movie_stride == 0 &&
           xT_now >= cfg.movie_endpoint_x_min &&
           xT_now < cfg.movie_endpoint_x_max
            abp_maybe_save_movie_path!(
                acc, sys;
                chain_id=chain_id,
                mcmc_step=mcmc_step,
                xT_now=xT_now,
                yT_now=yT_now,
                ymean_now=ymean_now,
                yint_now=yint_now,
                bias_value_now=bias_value_now,
                w_path=w_path,
                max_movie_paths=cfg.movie_max_trajectories,
                movie_path_time_thin=cfg.movie_path_time_thin,
            )
        end

        MonteCarloX.update!(rt_prod, bias_value_now)
        if mcmc_step % cfg.roundtrip_stride == 0
            push!(acc.rt_chain_ids, chain_id)
            push!(acc.rt_steps, mcmc_step)
            push!(acc.rt_counts, rt_prod.count)
            push!(acc.rt_values, bias_value_now)
        end

        abp_add_weighted_value!(acc.counts_x_T_biased, edges_x_T, xT_now, 1.0)
        if w_path > 0.0
            abp_add_weighted_value!(acc.counts_x_T_unbiased, edges_x_T, xT_now, w_path)
            acc.sum_w_all[1]  += w_path
            acc.sum_w2_all[1] += w_path^2
            acc.n_rew_all[1]  += 1
        end

        if mcmc_step % cfg.path_observation_stride == 0
            matching_path_windows = Int[]
            for (iw, spec) in enumerate(path_endpoint_specs)
                abp_matches_endpoint_spec(xT_now, spec) || continue
                push!(matching_path_windows, iw)
                acc.n_path_traj_biased[iw] += 1
                w_path > 0.0 && (acc.n_path_traj_unbiased[iw] += 1)
            end

            if !isempty(matching_path_windows)
                abp_add_whole_path_conditioned_windows!(
                    acc, sys, matching_path_windows, edges_path_x, edges_path_y, w_path;
                    path_time_stride=cfg.path_time_stride,
                    path_x_min=path_x_min,
                )
            end
        end

        for (iw, spec) in enumerate(endpoint_window_specs)
            abp_matches_endpoint_spec(xT_now, spec) || continue

            acc.n_biased_by_window[iw] += 1

            ok_y_b    = abp_add_weighted_value!(acc.counts_y_T_biased[iw],    edges_y_T,    yT_now,    1.0)
            ok_mean_b = abp_add_weighted_value!(acc.counts_y_mean_biased[iw], edges_y_mean, ymean_now, 1.0)
            ok_int_b  = abp_add_weighted_value!(acc.counts_y_int_biased[iw],  edges_y_int,  yint_now,  1.0)
            ok_xy_b   = abp_add_weighted_joint!(acc.counts_xy_T_biased[iw],   edges_x_T, edges_y_T, xT_now, yT_now, 1.0)

            if w_path > 0.0
                acc.n_rew_by_window[iw] += 1
                acc.sum_w_by_window[iw]  += w_path
                acc.sum_w2_by_window[iw] += w_path^2

                ok_y    = abp_add_weighted_value!(acc.counts_y_T_unbiased[iw],    edges_y_T,    yT_now,    w_path)
                ok_mean = abp_add_weighted_value!(acc.counts_y_mean_unbiased[iw], edges_y_mean, ymean_now, w_path)
                ok_int  = abp_add_weighted_value!(acc.counts_y_int_unbiased[iw],  edges_y_int,  yint_now,  w_path)
                ok_xy   = abp_add_weighted_joint!(acc.counts_xy_T_unbiased[iw],   edges_x_T, edges_y_T, xT_now, yT_now, w_path)

                ok_y    || (acc.n_y_T_out[iw]    += 1)
                ok_mean || (acc.n_y_mean_out[iw] += 1)
                ok_int  || (acc.n_y_int_out[iw]  += 1)
                ok_xy   || (acc.n_xy_T_out[iw]   += 1)
            else
                ok_y_b    || (acc.n_y_T_out[iw]    += 1)
                ok_mean_b || (acc.n_y_mean_out[iw] += 1)
                ok_int_b  || (acc.n_y_int_out[iw]  += 1)
                ok_xy_b   || (acc.n_xy_T_out[iw]   += 1)
            end

            if mcmc_step % cfg.prod_stride == 0
                abp_maybe_save_path!(
                    acc, sys, iw, xT_now, yT_now, ymean_now, yint_now, bias_value_now, w_path;
                    max_saved_paths_per_window_chain=max_saved_paths_per_window_chain,
                    saved_path_time_thin=cfg.saved_path_time_thin,
                )
            end
        end
    end

    # Always include the final production state, even when n_prod_steps is not
    # an exact multiple of roundtrip_stride.
    if n_prod_steps > 0 && (isempty(acc.rt_steps) || last(acc.rt_steps) != n_prod_steps)
        push!(acc.rt_chain_ids, chain_id)
        push!(acc.rt_steps, n_prod_steps)
        push!(acc.rt_counts, rt_prod.count)
        push!(acc.rt_values, obs(sys))
    end

    acc.rt_final_count[1] = rt_prod.count
    return acc
end
