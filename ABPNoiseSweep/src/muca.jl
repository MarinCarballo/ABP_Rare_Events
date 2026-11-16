# Recursive MUCA learning for one D value.

function abp_run_muca_one_case(cfg::ABPNoiseSweepConfig, D::Real)
    obs = endpoint_x
    bias_name = "endpoint_x"
    bias_label = "x(T)"

    abp = ABP(Dt=Float64(D), v=cfg.v, Dr=Float64(D), potential_lambda=cfg.potential_lambda)
    bins_bias = cfg.bias_min:cfg.dbias:cfg.bias_max
    edges_bias = collect(bins_bias)
    centers_bias = abp_centers_from_edges(edges_bias)

    @assert cfg.xT_min < cfg.xT_max
    @assert first(edges_bias) <= cfg.xT_min - cfg.xT_extension_margin
    @assert last(edges_bias)  >= cfg.xT_max + cfg.xT_extension_margin

    xT_min_extend = cfg.xT_min - cfg.xT_extension_margin
    xT_max_extend = cfg.xT_max + cfg.xT_extension_margin
    tail_temperature = abp_tail_temperature_for_D(D)
    n_iter_factor = abp_n_iter_factor_for_D(cfg, D)
    n_iter_effective = abp_n_iter_for_D(cfg, D)

    n_chains = Threads.nthreads()
    if cfg.save_muca_movie_paths && cfg.muca_movie_max_trajectories > 0 && cfg.movie_chain_id > n_chains
        error("movie_chain_id exceeds the number of MUCA chains ($n_chains).")
    end
    backend = ThreadsBackend(n_chains)

    pmuca_algs = [
        Multicanonical(
            Xoshiro(cfg.seed + 30_000 + 1_000_000 * round(Int, 10_000 * D) + 10 * i),
            BinnedObject(bins_bias, 0.0),
        )
        for i in 1:n_chains
    ]

    pmuca = ParallelMulticanonical(backend, pmuca_algs)

    pmuca_sys = [
        ABPTrajectory(
            Xoshiro(cfg.seed + 40_000 + 1_000_000 * round(Int, 10_000 * D) + 10 * i),
            abp,
            cfg.trajectory_T;
            dt=cfg.dt,
            x0=cfg.x0_vec,
            θ0=:uniform,
            potential_active=cfg.potential_active,
            integrator=cfg.integrator,
        )
        for i in 1:n_chains
    ]

    pmuca_rts = [Roundtrips(cfg.xT_min, cfg.xT_max) for _ in 1:n_chains]
    pmuca_moves, pmuca_move_weights = make_abp_moves(
        obs,
        bins_bias,
        cfg.potential_active;
        move_weights=cfg.move_weights,
        block_dxi=cfg.block_dxi,
        local_dxi=cfg.local_dxi,
        noise_proposal=cfg.noise_proposal,
        block_pcn_beta=cfg.block_pcn_beta,
        local_pcn_beta=cfg.local_pcn_beta,
    )

    on_root(pmuca) do i
        extend_muca_sides_tail_temperature!(
            algorithm(pmuca, i);
            xT_min_extend=xT_min_extend,
            xT_max_extend=xT_max_extend,
            tail_temperature_low=tail_temperature,
            tail_temperature_high=tail_temperature,
            extend_low=true,
            extend_high=true,
        )
    end
    distribute_logweight!(pmuca)

    iter_hist_values = Vector{Vector{Float64}}()
    iter_logw_values = Vector{Vector{Float64}}()
    iter_accept = Float64[]
    iter_flatness_maxmean = Float64[]
    iter_flatness_meanmin = Float64[]
    iter_roundtrips = Int[]
    iter_sampling_steps = Int[]
    iter_total_mcmc_steps = Int[]
    iter_roundtrips_per_sampling_step = Float64[]
    iter_steps_per_target_roundtrips = Float64[]
    iter_avg_roundtrips_per_chain = Float64[]
    
    n_sweeps_per_chain(i_iter) = max(
        1,
        round(Int, (cfg.n_iter_steps_per_iter * i_iter) / (n_iter_effective * n_chains)),
    )

    # Only the selected chain writes this vector. Capture is observational:
    # no random draws or modifications to trajectory/histogram state.
    muca_movie_paths = Any[]
    movie_acc = (movie_paths=muca_movie_paths,)
    planned_sampling_steps = sum(n_sweeps_per_chain(i) for i in 1:n_iter_effective)
    muca_movie_stride = abp_movie_effective_stride(
        cfg.muca_movie_stride, planned_sampling_steps, cfg.muca_movie_max_trajectories,
    )
    completed_sampling_steps = 0
    completed_mcmc_steps = 0
    if cfg.save_muca_movie_paths
        println("MUCA movie snapshots: chain=", cfg.movie_chain_id,
                " | effective sampling stride=", muca_movie_stride,
                " | max trajectories=", cfg.muca_movie_max_trajectories)
    end

    println()
    println("============================================================")
    println("MUCA case: D=", D, " Dt=Dr")
    println("trajectory_T=", cfg.trajectory_T, " | tail_temperature=", tail_temperature, " | slope=", -1/tail_temperature)
    println("potential lambda=", cfg.potential_lambda, " | k=0.3")
    println("D scaling reference=", cfg.D_scaling_reference,
            " | n_iter factor=", round(n_iter_factor; sigdigits=4),
            " | effective n_iter=", n_iter_effective)
    println("noise proposal=", cfg.noise_proposal,
            " | proposal Δξ block=", round(cfg.block_dxi; sigdigits=4),
            " | proposal Δξ local=", round(cfg.local_dxi; sigdigits=4),
            " | move weights=", cfg.move_weights)
    if cfg.noise_proposal === :pcn
        println("pCN beta (block/local) = ", cfg.block_pcn_beta, " / ", cfg.local_pcn_beta)
    end
    println("chains=", n_chains, " | bins=[", first(edges_bias), ", ", last(edges_bias), "] | dbias=", cfg.dbias)
        target_avg_roundtrips_per_chain = cfg.roundtrip_avg_target_fraction * n_chains
        println("early-stop target: average roundtrips per chain ≥ ", target_avg_roundtrips_per_chain,
            " for ", cfg.roundtrip_convergence_hits, " straight iterations")
    println("============================================================")

        prog = Progress(n_iter_effective; desc="MUCA D=$(D)...")
        stop_hits = 0
        stop_iteration = 0
    t_muca = @elapsed for i_iter in 1:n_iter_effective
        n_sweeps_i = n_sweeps_per_chain(i_iter)

        with_parallel(pmuca) do i, alg
            local sys = pmuca_sys[i]
            rt  = pmuca_rts[i]

            for _ in 1:cfg.n_therm_muca
                random_move!(sys, alg, pmuca_moves, pmuca_move_weights)
            end

            reset!(alg)
            reset!(rt)

            for iteration_step in 1:n_sweeps_i
                random_move!(sys, alg, pmuca_moves, pmuca_move_weights)
                MonteCarloX.update!(rt, obs(sys))
                sampling_step = completed_sampling_steps + iteration_step
                if cfg.save_muca_movie_paths && i == cfg.movie_chain_id &&
                   (sampling_step - 1) % muca_movie_stride == 0 &&
                   cfg.movie_endpoint_x_min <= obs(sys) < cfg.movie_endpoint_x_max
                    abp_maybe_save_movie_path!(
                        movie_acc, sys;
                        chain_id=i,
                        mcmc_step=completed_mcmc_steps + cfg.n_therm_muca + iteration_step,
                        xT_now=endpoint_x(sys), yT_now=endpoint_y(sys),
                        ymean_now=mean_y(sys), yint_now=path_y_int(sys),
                        bias_value_now=obs(sys), w_path=NaN,
                        max_movie_paths=cfg.muca_movie_max_trajectories,
                        movie_path_time_thin=cfg.movie_path_time_thin,
                        phase=:muca, iteration=i_iter,
                        iteration_step=iteration_step, sampling_step=sampling_step,
                    )
                end
            end
        end

        completed_sampling_steps += n_sweeps_i
        completed_mcmc_steps += cfg.n_therm_muca + n_sweeps_i

        merge_histograms!(pmuca)

        on_root(pmuca) do i
            root_alg = algorithm(pmuca, i)
            root_ens = ensemble(root_alg)

            MonteCarloX.update!(root_ens; mode=:recursive)

            extend_muca_sides_tail_temperature!(
                root_alg;
                xT_min_extend=xT_min_extend,
                xT_max_extend=xT_max_extend,
                tail_temperature_low=tail_temperature,
                tail_temperature_high=tail_temperature,
                extend_low=true,
                extend_high=true,
            )

            push!(iter_hist_values, copy(Float64.(root_ens.histogram.values)))
            push!(iter_logw_values, copy(Float64.(logweight(root_alg).values)))
            push!(iter_accept, mean([acceptance_rate(algorithm(pmuca, j)) for j in 1:n_chains]))
            push!(iter_flatness_maxmean, flatness(root_ens.histogram, cfg.xT_min, cfg.xT_max; criterion=:max_over_mean))
            push!(iter_flatness_meanmin, flatness(root_ens.histogram, cfg.xT_min, cfg.xT_max; criterion=:mean_over_min))
            push!(iter_roundtrips, sum(rt.count for rt in pmuca_rts))
        end

        # Normalize roundtrips by the actual amount of work in this iteration..
        sampling_steps_i = n_sweeps_i * n_chains # sampling steps across all chains in this iteration
        total_steps_i = sampling_steps_i + cfg.n_therm_muca * n_chains
        rt_i = iter_roundtrips[end] # roundtrips in this iteration across all chains
        avg_roundtrips_i = rt_i / n_chains
        rate_i = sampling_steps_i > 0 ? rt_i / sampling_steps_i : NaN # normalized roundtrip rate per sampling step
        steps_target_i = (isfinite(rate_i) && rate_i > 0) ? cfg.roundtrip_target / rate_i : Inf

        push!(iter_sampling_steps, sampling_steps_i)
        push!(iter_total_mcmc_steps, total_steps_i)
        push!(iter_roundtrips_per_sampling_step, rate_i)
        push!(iter_steps_per_target_roundtrips, steps_target_i)
        push!(iter_avg_roundtrips_per_chain, avg_roundtrips_i)

        if isfinite(avg_roundtrips_i) && avg_roundtrips_i >= target_avg_roundtrips_per_chain
            stop_hits += 1
        else
            stop_hits = 0
        end
        if stop_hits >= cfg.roundtrip_convergence_hits
            stop_iteration = i_iter
            println(
                "Stopping MUCA early at iteration ", i_iter,
                " | avg roundtrips/chain=", round(avg_roundtrips_i; digits=4),
                " | target=", round(target_avg_roundtrips_per_chain; digits=4),
                " | consecutive hits=", stop_hits,
            )
            next!(prog)
            break
        end

        distribute_logweight!(pmuca)
        next!(prog)

        if i_iter % 10 == 0
            println(
                "Completed MUCA iteration ", i_iter, "/", n_iter_effective,
                " | accept=", round(iter_accept[end]; digits=4),
                " | flatness max/mean=", round(iter_flatness_maxmean[end]; digits=3),
                " | roundtrips=", iter_roundtrips[end],
                " | steps/", cfg.roundtrip_target, " RT≈", round(iter_steps_per_target_roundtrips[end]; sigdigits=4),
            )
        end
    end

    if stop_iteration == 0
        stop_iteration = length(iter_avg_roundtrips_per_chain)
    end

    if cfg.save_muca_movie_paths && cfg.movie_chain_id <= n_chains
        final_movie_sys = pmuca_sys[cfg.movie_chain_id]
        if cfg.movie_endpoint_x_min <= obs(final_movie_sys) < cfg.movie_endpoint_x_max
            abp_maybe_save_movie_path!(
                movie_acc, final_movie_sys;
                chain_id=cfg.movie_chain_id, mcmc_step=completed_mcmc_steps,
                xT_now=endpoint_x(final_movie_sys), yT_now=endpoint_y(final_movie_sys),
                ymean_now=mean_y(final_movie_sys), yint_now=path_y_int(final_movie_sys),
                bias_value_now=obs(final_movie_sys), w_path=NaN,
                max_movie_paths=cfg.muca_movie_max_trajectories,
                movie_path_time_thin=cfg.movie_path_time_thin,
                phase=:muca, iteration=stop_iteration,
                iteration_step=n_sweeps_per_chain(stop_iteration),
                sampling_step=completed_sampling_steps, replace_last=true,
            )
        end
    end

    alg_final = Ref{Any}()
    on_root(pmuca) do i
        alg_final[] = deepcopy(algorithm(pmuca, i))
    end

    n_muca_total = sum(n_sweeps_per_chain(i) * n_chains + cfg.n_therm_muca * n_chains for i in 1:stop_iteration)
    roundtrip_steps_summary = abp_roundtrip_steps_summary(
        iter_steps_per_target_roundtrips;
        window=cfg.roundtrip_target,
        rtol=0.0,
    )
    roundtrip_stop_summary = abp_roundtrip_average_stop_summary(
        iter_avg_roundtrips_per_chain;
        target_avg_roundtrips_per_chain=target_avg_roundtrips_per_chain,
        hits_required=cfg.roundtrip_convergence_hits,
    )

    println("MUCA complete for D=", D,
            " | elapsed=", round(t_muca; digits=1), "s | total sweeps≈", round(Int, n_muca_total),
            " | final roundtrips=", iter_roundtrips[end],
            " | final avg roundtrips/chain=", round(iter_avg_roundtrips_per_chain[end]; digits=4))
    if roundtrip_stop_summary.stopped_early
        println("Early-stop condition met at iteration ", roundtrip_stop_summary.stop_iteration,
                " after ", roundtrip_stop_summary.consecutive_hits, " consecutive hits.")
    else
        println("Early-stop condition was not reached.")
    end

    return (
        alg = alg_final[],
        abp = abp,
        obs = obs,
        bias_name = bias_name,
        bias_label = bias_label,
        bins_bias = bins_bias,
        edges_bias = edges_bias,
        centers_bias = centers_bias,
        iter_hist_values = iter_hist_values,
        iter_logw_values = iter_logw_values,
        iter_accept = iter_accept,
        iter_flatness_maxmean = iter_flatness_maxmean,
        iter_flatness_meanmin = iter_flatness_meanmin,
        iter_roundtrips = iter_roundtrips,
        iter_sampling_steps = iter_sampling_steps,
        iter_total_mcmc_steps = iter_total_mcmc_steps,
        iter_roundtrips_per_sampling_step = iter_roundtrips_per_sampling_step,
        iter_steps_per_target_roundtrips = iter_steps_per_target_roundtrips,
        iter_avg_roundtrips_per_chain = iter_avg_roundtrips_per_chain,
        roundtrip_steps_summary = roundtrip_steps_summary,
        roundtrip_stop_summary = roundtrip_stop_summary,
        tail_temperature = tail_temperature,
        D_scaling_reference = cfg.D_scaling_reference,
        n_iter_factor = n_iter_factor,
        n_iter_effective = n_iter_effective,
        block_dxi = cfg.block_dxi,
        local_dxi = cfg.local_dxi,
        xT_min_extend = xT_min_extend,
        xT_max_extend = xT_max_extend,
        n_chains = n_chains,
        movie_paths = muca_movie_paths,
        movie_effective_stride = muca_movie_stride,
        n_muca_total = n_muca_total,
        t_muca = t_muca,
    )
end

# -----------------------------
# Production uses the learned MUCA bias for one D value.
# -----------------------------

