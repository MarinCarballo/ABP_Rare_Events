# Production histogram accumulators and path-saving helpers.

function abp_make_prod_accumulator(
    edges_x_T,
    edges_y_T,
    edges_y_mean,
    edges_y_int,
    edges_path_x,
    edges_path_y,
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

function abp_add_whole_path_conditioned_windows!(
    acc,
    sys,
    path_window_indices,
    edges_path_x,
    edges_path_y,
    unbias_weight::Real;
    path_time_stride::Int=1,
    path_x_min::Real=-0.6,
)
    # The endpoint condition is evaluated before this function is called.
    # The trajectory is scanned once and accumulated into every matching endpoint window.
    use_unbiased = isfinite(unbias_weight) && unbias_weight > 0.0

    @inbounds for k in 1:path_time_stride:length(sys.xs)
        x = sys.xs[k][1]
        y = sys.xs[k][2]
        x <= path_x_min && continue

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
