# Recursive MUCA learning for one D value.

function abp_run_muca_one_case(cfg::ABPNoiseSweepConfig, D::Real)
    obs = endpoint_x
    bias_name = "endpoint_x"
    bias_label = "x(T)"

    abp = ABP(Dt=Float64(D), v=cfg.v, Dr=Float64(D))
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

    n_sweeps_per_chain(i_iter) = max(
        1,
        round(Int, (cfg.n_iter_steps_per_iter * i_iter) / (n_iter_effective * n_chains)),
    )

    println()
    println("============================================================")
    println("MUCA case: D=", D, " Dt=Dr")
    println("trajectory_T=", cfg.trajectory_T, " | tail_temperature=", tail_temperature, " | slope=", -1/tail_temperature)
    println("D scaling reference=", cfg.D_scaling_reference,
            " | n_iter factor=", round(n_iter_factor; sigdigits=4),
            " | effective n_iter=", n_iter_effective)
    println("proposal Δξ block=", round(cfg.block_dxi; sigdigits=4),
            " | proposal Δξ local=", round(cfg.local_dxi; sigdigits=4),
            " | move weights=", cfg.move_weights)
    println("chains=", n_chains, " | bins=[", first(edges_bias), ", ", last(edges_bias), "] | dbias=", cfg.dbias)
    println("============================================================")

    prog = Progress(n_iter_effective; desc="MUCA D=$(D)...")
    t_muca = @elapsed for i_iter in 1:n_iter_effective
        n_sweeps_i = n_sweeps_per_chain(i_iter)

        with_parallel(pmuca) do i, alg
            sys = pmuca_sys[i]
            rt  = pmuca_rts[i]

            for _ in 1:cfg.n_therm_muca
                random_move!(sys, alg, pmuca_moves, pmuca_move_weights)
            end

            reset!(alg)
            reset!(rt)

            for _ in 1:n_sweeps_i
                random_move!(sys, alg, pmuca_moves, pmuca_move_weights)
                MonteCarloX.update!(rt, obs(sys))
            end
        end

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
        rate_i = sampling_steps_i > 0 ? rt_i / sampling_steps_i : NaN # normalized roundtrip rate per sampling step
        steps_target_i = (isfinite(rate_i) && rate_i > 0) ? cfg.roundtrip_target / rate_i : Inf

        push!(iter_sampling_steps, sampling_steps_i)
        push!(iter_total_mcmc_steps, total_steps_i)
        push!(iter_roundtrips_per_sampling_step, rate_i)
        push!(iter_steps_per_target_roundtrips, steps_target_i)

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

    alg_final = Ref{Any}()
    on_root(pmuca) do i
        alg_final[] = deepcopy(algorithm(pmuca, i))
    end

    n_muca_total = sum(n_sweeps_per_chain(i) * n_chains + cfg.n_therm_muca * n_chains for i in 1:n_iter_effective)
    roundtrip_steps_summary = abp_roundtrip_steps_summary(
        iter_steps_per_target_roundtrips;
        window=cfg.roundtrip_convergence_window,
        rtol=cfg.roundtrip_convergence_rtol,
    )

    println("MUCA complete for D=", D,
            " | elapsed=", round(t_muca; digits=1), "s | total sweeps≈", round(Int, n_muca_total),
            " | final roundtrips=", iter_roundtrips[end])
    println("Roundtrip-normalized estimate for ", cfg.roundtrip_target, " roundtrips: ",
            round(roundtrip_steps_summary.steps_estimate; sigdigits=5),
            " sampling moves | converged=", roundtrip_steps_summary.converged,
            " | rel_half_range=", round(roundtrip_steps_summary.rel_half_range; digits=3))

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
        roundtrip_steps_summary = roundtrip_steps_summary,
        tail_temperature = tail_temperature,
        D_scaling_reference = cfg.D_scaling_reference,
        n_iter_factor = n_iter_factor,
        n_iter_effective = n_iter_effective,
        block_dxi = cfg.block_dxi,
        local_dxi = cfg.local_dxi,
        xT_min_extend = xT_min_extend,
        xT_max_extend = xT_max_extend,
        n_chains = n_chains,
        n_muca_total = n_muca_total,
        t_muca = t_muca,
    )
end

# -----------------------------
# Production uses the learned MUCA bias for one D value.
# -----------------------------

