# Production histogram accumulators and path-saving helpers.

function abp_make_prod_accumulator(
    edges_x_T,
    edges_y_T,
    edges_y_mean,
    edges_y_int,
    edges_path_x,
    edges_path_y,
    edges_path_theta,
    n_windows,
    n_path_windows,
)
    return (
        counts_x_T_biased   = zeros(Float64, length(edges_x_T) - 1),
        counts_x_T_unbiased = zeros(Float64, length(edges_x_T) - 1),

        counts_y_T_biased      = [zeros(Float64, length(edges_y_T) - 1) for _ in 1:n_windows],
        counts_y_T_unbiased    = [zeros(Float64, length(edges_y_T) - 1) for _ in 1:n_windows],
        counts_y_mean_biased   = [zeros(Float64, length(edges_y_mean) - 1) for _ in 1:n_windows],
        counts_y_mean_unbiased = [zeros(Float64, length(edges_y_mean) - 1) for _ in 1:n_windows],
        counts_y_int_biased    = [zeros(Float64, length(edges_y_int) - 1) for _ in 1:n_windows],
        counts_y_int_unbiased  = [zeros(Float64, length(edges_y_int) - 1) for _ in 1:n_windows],
        counts_xy_T_biased     = [zeros(Float64, length(edges_x_T) - 1, length(edges_y_T) - 1) for _ in 1:n_windows],
        counts_xy_T_unbiased   = [zeros(Float64, length(edges_x_T) - 1, length(edges_y_T) - 1) for _ in 1:n_windows],

        # Whole-trajectory occupation histograms for several endpoint conditions.
        counts_path_y_biased = [
            zeros(Float64, length(edges_path_y) - 1) for _ in 1:n_path_windows
        ],
        counts_path_y_unbiased = [
            zeros(Float64, length(edges_path_y) - 1) for _ in 1:n_path_windows
        ],
        counts_path_xy_biased = [
            zeros(Float64, length(edges_path_x) - 1, length(edges_path_y) - 1)
            for _ in 1:n_path_windows
        ],
        counts_path_xy_unbiased = [
            zeros(Float64, length(edges_path_x) - 1, length(edges_path_y) - 1)
            for _ in 1:n_path_windows
        ],

        # Joint occupation histogram P(x(t), theta(t) | x(T)>0.5).  Theta is
        # wrapped to [0, 2pi).  Unlike path x-y, this is intentionally stored
        # only for the route-analysis endpoint condition to limit memory.
        counts_path_xtheta_biased = zeros(
            Float64, length(edges_path_x) - 1, length(edges_path_theta) - 1,
        ),
        counts_path_xtheta_unbiased = zeros(
            Float64, length(edges_path_x) - 1, length(edges_path_theta) - 1,
        ),

        n_path_traj_biased   = zeros(Int, n_path_windows),
        n_path_traj_unbiased = zeros(Int, n_path_windows),

        sum_w_by_window    = zeros(Float64, n_windows),
        sum_w2_by_window   = zeros(Float64, n_windows),
        n_biased_by_window = zeros(Int, n_windows),
        n_rew_by_window    = zeros(Int, n_windows),

        n_y_T_out    = zeros(Int, n_windows),
        n_y_mean_out = zeros(Int, n_windows),
        n_y_int_out  = zeros(Int, n_windows),
        n_xy_T_out   = zeros(Int, n_windows),

        n_path_y_out_biased    = zeros(Int, n_path_windows),
        n_path_y_out_unbiased  = zeros(Int, n_path_windows),
        n_path_xy_out_biased   = zeros(Int, n_path_windows),
        n_path_xy_out_unbiased = zeros(Int, n_path_windows),
        n_path_xtheta_out_biased   = [0],
        n_path_xtheta_out_unbiased = [0],

        # Active-work sufficient statistics, binned by the first
        # left-to-right crossing ordinate m = y(tau_c).  These retain enough
        # information for the notebooks to choose route cuts after production.
        active_work_mass_biased       = zeros(Float64, length(edges_y_T) - 1),
        active_work_mass_unbiased     = zeros(Float64, length(edges_y_T) - 1),
        active_work_mass2_unbiased    = zeros(Float64, length(edges_y_T) - 1),
        active_work_sum_biased        = zeros(Float64, length(edges_y_T) - 1),
        active_work_sum2_biased       = zeros(Float64, length(edges_y_T) - 1),
        active_work_sum_unbiased      = zeros(Float64, length(edges_y_T) - 1),
        active_work_sum2_unbiased     = zeros(Float64, length(edges_y_T) - 1),
        crossing_time_sum_biased      = zeros(Float64, length(edges_y_T) - 1),
        crossing_time_sum_unbiased    = zeros(Float64, length(edges_y_T) - 1),
        n_active_work_observations    = [0],
        n_active_work_no_crossing     = [0],
        n_active_work_crossing_y_out  = [0],

        sum_w_all  = [0.0],
        sum_w2_all = [0.0],
        n_rew_all  = [0],

        # Production roundtrip traces. Counts are cumulative within each chain.
        rt_chain_ids = Int[],
        rt_steps     = Int[],
        rt_counts    = Int[],
        rt_values    = Float64[],
        rt_final_count = [0],

        saved_paths = [Any[] for _ in 1:n_windows],

        # Raw path snapshots for movie diagnostics. Each entry is one production
        # Markov state trajectory, thinned in physical time.
        movie_paths = Any[],
    )
end

function abp_merge_prod_accumulators!(a, b; max_saved_paths_per_window::Int, max_movie_paths::Int=typemax(Int))
    a.counts_x_T_biased   .+= b.counts_x_T_biased
    a.counts_x_T_unbiased .+= b.counts_x_T_unbiased

    for i in eachindex(a.counts_y_T_biased)
        a.counts_y_T_biased[i]      .+= b.counts_y_T_biased[i]
        a.counts_y_T_unbiased[i]    .+= b.counts_y_T_unbiased[i]
        a.counts_y_mean_biased[i]   .+= b.counts_y_mean_biased[i]
        a.counts_y_mean_unbiased[i] .+= b.counts_y_mean_unbiased[i]
        a.counts_y_int_biased[i]    .+= b.counts_y_int_biased[i]
        a.counts_y_int_unbiased[i]  .+= b.counts_y_int_unbiased[i]
        a.counts_xy_T_biased[i]     .+= b.counts_xy_T_biased[i]
        a.counts_xy_T_unbiased[i]   .+= b.counts_xy_T_unbiased[i]
    end

    for i in eachindex(a.counts_path_y_biased)
        a.counts_path_y_biased[i]   .+= b.counts_path_y_biased[i]
        a.counts_path_y_unbiased[i] .+= b.counts_path_y_unbiased[i]
        a.counts_path_xy_biased[i]  .+= b.counts_path_xy_biased[i]
        a.counts_path_xy_unbiased[i] .+= b.counts_path_xy_unbiased[i]
    end
    a.counts_path_xtheta_biased   .+= b.counts_path_xtheta_biased
    a.counts_path_xtheta_unbiased .+= b.counts_path_xtheta_unbiased

    a.n_path_traj_biased   .+= b.n_path_traj_biased
    a.n_path_traj_unbiased .+= b.n_path_traj_unbiased

    a.sum_w_by_window    .+= b.sum_w_by_window
    a.sum_w2_by_window   .+= b.sum_w2_by_window
    a.n_biased_by_window .+= b.n_biased_by_window
    a.n_rew_by_window    .+= b.n_rew_by_window

    a.n_y_T_out    .+= b.n_y_T_out
    a.n_y_mean_out .+= b.n_y_mean_out
    a.n_y_int_out  .+= b.n_y_int_out
    a.n_xy_T_out   .+= b.n_xy_T_out

    a.n_path_y_out_biased    .+= b.n_path_y_out_biased
    a.n_path_y_out_unbiased  .+= b.n_path_y_out_unbiased
    a.n_path_xy_out_biased   .+= b.n_path_xy_out_biased
    a.n_path_xy_out_unbiased .+= b.n_path_xy_out_unbiased
    a.n_path_xtheta_out_biased[1]   += b.n_path_xtheta_out_biased[1]
    a.n_path_xtheta_out_unbiased[1] += b.n_path_xtheta_out_unbiased[1]

    a.active_work_mass_biased    .+= b.active_work_mass_biased
    a.active_work_mass_unbiased  .+= b.active_work_mass_unbiased
    a.active_work_mass2_unbiased .+= b.active_work_mass2_unbiased
    a.active_work_sum_biased     .+= b.active_work_sum_biased
    a.active_work_sum2_biased    .+= b.active_work_sum2_biased
    a.active_work_sum_unbiased   .+= b.active_work_sum_unbiased
    a.active_work_sum2_unbiased  .+= b.active_work_sum2_unbiased
    a.crossing_time_sum_biased   .+= b.crossing_time_sum_biased
    a.crossing_time_sum_unbiased .+= b.crossing_time_sum_unbiased
    a.n_active_work_observations[1]   += b.n_active_work_observations[1]
    a.n_active_work_no_crossing[1]    += b.n_active_work_no_crossing[1]
    a.n_active_work_crossing_y_out[1] += b.n_active_work_crossing_y_out[1]

    a.sum_w_all[1]  += b.sum_w_all[1]
    a.sum_w2_all[1] += b.sum_w2_all[1]
    a.n_rew_all[1]  += b.n_rew_all[1]

    append!(a.rt_chain_ids, b.rt_chain_ids)
    append!(a.rt_steps, b.rt_steps)
    append!(a.rt_counts, b.rt_counts)
    append!(a.rt_values, b.rt_values)
    a.rt_final_count[1] += b.rt_final_count[1]

    for i in eachindex(a.saved_paths)
        append!(a.saved_paths[i], b.saved_paths[i])
        if length(a.saved_paths[i]) > max_saved_paths_per_window
            resize!(a.saved_paths[i], max_saved_paths_per_window)
        end
    end

    append!(a.movie_paths, b.movie_paths)
    if length(a.movie_paths) > max_movie_paths
        resize!(a.movie_paths, max_movie_paths)
    end

    return a
end

function abp_add_active_work_crossing!(
    acc,
    crossing_y_edges,
    route_metrics,
    unbias_weight::Real,
)
    acc.n_active_work_observations[1] += 1

    if !route_metrics.has_crossing
        acc.n_active_work_no_crossing[1] += 1
        return false
    end

    ibin = abp_bin_index_from_edges(route_metrics.crossing_y, crossing_y_edges)
    if ibin === nothing
        acc.n_active_work_crossing_y_out[1] += 1
        return false
    end

    W_active = route_metrics.active_work
    tau_cross = route_metrics.crossing_time

    acc.active_work_mass_biased[ibin] += 1.0
    acc.active_work_sum_biased[ibin] += W_active
    acc.active_work_sum2_biased[ibin] += W_active^2
    acc.crossing_time_sum_biased[ibin] += tau_cross

    if isfinite(unbias_weight) && unbias_weight > 0.0
        acc.active_work_mass_unbiased[ibin] += unbias_weight
        acc.active_work_mass2_unbiased[ibin] += unbias_weight^2
        acc.active_work_sum_unbiased[ibin] += unbias_weight * W_active
        acc.active_work_sum2_unbiased[ibin] += unbias_weight * W_active^2
        acc.crossing_time_sum_unbiased[ibin] += unbias_weight * tau_cross
    end

    return true
end

function abp_add_whole_path_conditioned_windows!(
    acc,
    sys,
    path_window_indices,
    edges_path_x,
    edges_path_y,
    edges_path_theta,
    theta_path_window::Int,
    unbias_weight::Real;
    path_time_stride::Int=1,
    path_x_min::Real=-0.6,
    compute_route_metrics::Bool=false,
    route_crossing_x::Real=0.0,
)
    # The endpoint condition is evaluated before this function is called.
    # The trajectory is scanned once and accumulated into every matching endpoint window.
    use_unbiased = isfinite(unbias_weight) && unbias_weight > 0.0
    accumulate_theta = theta_path_window in path_window_indices

    W_active = 0.0
    crossing_y = NaN
    crossing_time = NaN
    has_crossing = false
    x_cross = Float64(route_crossing_x)

    if compute_route_metrics
        # Work/crossing use every integration step.  The occupation histogram
        # is updated in the same pass at its original time stride.
        @inbounds for k in eachindex(sys.xs)
            if k < length(sys.xs)
                r0 = sys.xs[k]
                r1 = sys.xs[k + 1]
                dr = r1 - r0
                theta_mid = 0.5 * (sys.θs[k] + sys.θs[k + 1])
                W_active += sys.v * (
                    cos(theta_mid) * dr[1] + sin(theta_mid) * dr[2]
                )

                if !has_crossing && r0[1] < x_cross <= r1[1]
                    alpha = (x_cross - r0[1]) / (r1[1] - r0[1])
                    crossing_y = r0[2] + alpha * (r1[2] - r0[2])
                    crossing_time = ((k - 1) + alpha) * sys.dt
                    has_crossing = true
                end
            end

            (k - 1) % path_time_stride == 0 || continue
            abp_add_path_point_conditioned_windows!(
                acc, sys.xs[k], sys.θs[k], path_window_indices, edges_path_x,
                edges_path_y, edges_path_theta, accumulate_theta,
                unbias_weight, use_unbiased, path_x_min,
            )
        end
    else
        # Preserve the cheap, time-thinned occupation scan on observations
        # where active work is not measured.
        @inbounds for k in 1:path_time_stride:length(sys.xs)
            abp_add_path_point_conditioned_windows!(
                acc, sys.xs[k], sys.θs[k], path_window_indices, edges_path_x,
                edges_path_y, edges_path_theta, accumulate_theta,
                unbias_weight, use_unbiased, path_x_min,
            )
        end
    end

    return compute_route_metrics ? (
        active_work = W_active,
        crossing_y = crossing_y,
        crossing_time = crossing_time,
        has_crossing = has_crossing,
    ) : nothing
end

@inline function abp_add_path_point_conditioned_windows!(
    acc,
    position,
    theta,
    path_window_indices,
    edges_path_x,
    edges_path_y,
    edges_path_theta,
    accumulate_theta::Bool,
    unbias_weight::Real,
    use_unbiased::Bool,
    path_x_min::Real,
)
    x = position[1]
    y = position[2]
    x <= path_x_min && return nothing

    for iw in path_window_indices
        ok_y_b = abp_add_weighted_value!(
            acc.counts_path_y_biased[iw], edges_path_y, y, 1.0,
        )
        ok_xy_b = abp_add_weighted_joint!(
            acc.counts_path_xy_biased[iw], edges_path_x, edges_path_y, x, y, 1.0,
        )
        ok_y_b  || (acc.n_path_y_out_biased[iw] += 1)
        ok_xy_b || (acc.n_path_xy_out_biased[iw] += 1)

        if use_unbiased
            ok_y_u = abp_add_weighted_value!(
                acc.counts_path_y_unbiased[iw], edges_path_y, y, unbias_weight,
            )
            ok_xy_u = abp_add_weighted_joint!(
                acc.counts_path_xy_unbiased[iw], edges_path_x, edges_path_y, x, y, unbias_weight,
            )
            ok_y_u  || (acc.n_path_y_out_unbiased[iw] += 1)
            ok_xy_u || (acc.n_path_xy_out_unbiased[iw] += 1)
        end
    end

    if accumulate_theta
        theta_wrapped = mod2pi(theta)
        ok_xtheta_b = abp_add_weighted_joint!(
            acc.counts_path_xtheta_biased,
            edges_path_x,
            edges_path_theta,
            x,
            theta_wrapped,
            1.0,
        )
        ok_xtheta_b || (acc.n_path_xtheta_out_biased[1] += 1)

        if use_unbiased
            ok_xtheta_u = abp_add_weighted_joint!(
                acc.counts_path_xtheta_unbiased,
                edges_path_x,
                edges_path_theta,
                x,
                theta_wrapped,
                unbias_weight,
            )
            ok_xtheta_u || (acc.n_path_xtheta_out_unbiased[1] += 1)
        end
    end

    return nothing
end

function abp_maybe_save_path!(
    acc,
    sys,
    iw::Int,
    xT_now,
    yT_now,
    ymean_now,
    yint_now,
    bias_value_now,
    w_path;
    max_saved_paths_per_window_chain::Int,
    saved_path_time_thin::Int,
)
    length(acc.saved_paths[iw]) >= max_saved_paths_per_window_chain && return nothing

    inds = 1:saved_path_time_thin:length(sys.xs)
    push!(acc.saved_paths[iw], (
        xs = copy(sys.xs[inds]),
        thetas = mod2pi.(sys.θs[inds]),
        theta0 = mod2pi(sys.θs[1]),
        endpoint_x = xT_now,
        endpoint_y = yT_now,
        y_mean = ymean_now,
        y_int = yint_now,
        bias_value = bias_value_now,
        unbias_weight_shifted = w_path,
    ))

    return nothing
end

# -----------------------------


function abp_maybe_save_movie_path!(
    acc,
    sys;
    chain_id::Int,
    mcmc_step::Int,
    xT_now,
    yT_now,
    ymean_now,
    yint_now,
    bias_value_now,
    w_path,
    max_movie_paths::Int,
    movie_path_time_thin::Int,
)
    max_movie_paths <= 0 && return nothing
    length(acc.movie_paths) >= max_movie_paths && return nothing

    inds = 1:movie_path_time_thin:length(sys.xs)
    push!(acc.movie_paths, (
        movie_id = length(acc.movie_paths) + 1,
        chain_id = chain_id,
        mcmc_step = mcmc_step,
        xs = copy(sys.xs[inds]),
        thetas = copy(sys.θs[inds]),
        endpoint_x = xT_now,
        endpoint_y = yT_now,
        y_mean = ymean_now,
        y_int = yint_now,
        bias_value = bias_value_now,
        unbias_weight_shifted = w_path,
    ))

    return nothing
end
