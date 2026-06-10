# Production histogram accumulators and path-saving helpers.

function abp_make_prod_accumulator(edges_x_T, edges_y_T, edges_y_mean, edges_y_int, edges_path_x, edges_path_y, n_windows)
    return (
        counts_x_T_biased   = zeros(Float64, length(edges_x_T) - 1),
        counts_x_T_unbiased = zeros(Float64, length(edges_x_T) - 1),

        counts_y_T_biased    = [zeros(Float64, length(edges_y_T) - 1)    for _ in 1:n_windows],
        counts_y_T_unbiased  = [zeros(Float64, length(edges_y_T) - 1)    for _ in 1:n_windows],
        counts_y_mean_biased   = [zeros(Float64, length(edges_y_mean) - 1) for _ in 1:n_windows],
        counts_y_mean_unbiased = [zeros(Float64, length(edges_y_mean) - 1) for _ in 1:n_windows],
        counts_y_int_biased    = [zeros(Float64, length(edges_y_int) - 1)  for _ in 1:n_windows],
        counts_y_int_unbiased  = [zeros(Float64, length(edges_y_int) - 1)  for _ in 1:n_windows],
        counts_xy_T_biased   = [zeros(Float64, length(edges_x_T) - 1, length(edges_y_T) - 1) for _ in 1:n_windows],
        counts_xy_T_unbiased = [zeros(Float64, length(edges_x_T) - 1, length(edges_y_T) - 1) for _ in 1:n_windows],

        # Whole-trajectory, endpoint-conditioned histograms for x(T)>0.
        # The condition is x(T)>0; every time point of that trajectory is counted.
        counts_path_y_pos_biased   = zeros(Float64, length(edges_path_y) - 1),
        counts_path_y_pos_unbiased = zeros(Float64, length(edges_path_y) - 1),
        counts_path_xy_pos_biased   = zeros(Float64, length(edges_path_x) - 1, length(edges_path_y) - 1),
        counts_path_xy_pos_unbiased = zeros(Float64, length(edges_path_x) - 1, length(edges_path_y) - 1),

        sum_w_by_window   = zeros(Float64, n_windows),
        sum_w2_by_window  = zeros(Float64, n_windows),
        n_biased_by_window = zeros(Int, n_windows),
        n_rew_by_window    = zeros(Int, n_windows),

        n_y_T_out    = zeros(Int, n_windows),
        n_y_mean_out = zeros(Int, n_windows),
        n_y_int_out  = zeros(Int, n_windows),
        n_xy_T_out   = zeros(Int, n_windows),

        n_path_y_pos_out_biased    = [0],
        n_path_y_pos_out_unbiased  = [0],
        n_path_xy_pos_out_biased   = [0],
        n_path_xy_pos_out_unbiased = [0],

        sum_w_all  = [0.0],
        sum_w2_all = [0.0],
        n_rew_all  = [0],

        rt_steps  = Int[],
        rt_counts = Int[],
        rt_values = Float64[],
        rt_final_count = [0],

        saved_paths = [Any[] for _ in 1:n_windows],
    )
end

function abp_merge_prod_accumulators!(a, b; max_saved_paths_per_window::Int)
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

    a.counts_path_y_pos_biased   .+= b.counts_path_y_pos_biased
    a.counts_path_y_pos_unbiased .+= b.counts_path_y_pos_unbiased
    a.counts_path_xy_pos_biased   .+= b.counts_path_xy_pos_biased
    a.counts_path_xy_pos_unbiased .+= b.counts_path_xy_pos_unbiased

    a.sum_w_by_window   .+= b.sum_w_by_window
    a.sum_w2_by_window  .+= b.sum_w2_by_window
    a.n_biased_by_window .+= b.n_biased_by_window
    a.n_rew_by_window    .+= b.n_rew_by_window

    a.n_y_T_out    .+= b.n_y_T_out
    a.n_y_mean_out .+= b.n_y_mean_out
    a.n_y_int_out  .+= b.n_y_int_out
    a.n_xy_T_out   .+= b.n_xy_T_out

    a.n_path_y_pos_out_biased[1]    += b.n_path_y_pos_out_biased[1]
    a.n_path_y_pos_out_unbiased[1]  += b.n_path_y_pos_out_unbiased[1]
    a.n_path_xy_pos_out_biased[1]   += b.n_path_xy_pos_out_biased[1]
    a.n_path_xy_pos_out_unbiased[1] += b.n_path_xy_pos_out_unbiased[1]

    a.sum_w_all[1]  += b.sum_w_all[1]
    a.sum_w2_all[1] += b.sum_w2_all[1]
    a.n_rew_all[1]  += b.n_rew_all[1]

    # Roundtrip traces from independent production chains are stored together as diagnostics.
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

    return a
end

function abp_add_whole_path_conditioned_pos!(acc, sys, edges_path_x, edges_path_y, weight::Real; path_time_stride::Int=1, reweighted::Bool=true)
    # Endpoint condition is handled before this function is called.
    # This function counts the whole trajectory, not only the endpoint.
    @inbounds for k in 1:path_time_stride:length(sys.xs)
        x = sys.xs[k][1]
        y = sys.xs[k][2]
        if reweighted
            ok_y  = abp_add_weighted_value!(acc.counts_path_y_pos_unbiased, edges_path_y, y, weight)
            ok_xy = abp_add_weighted_joint!(acc.counts_path_xy_pos_unbiased, edges_path_x, edges_path_y, x, y, weight)
            ok_y  || (acc.n_path_y_pos_out_unbiased[1] += 1)
            ok_xy || (acc.n_path_xy_pos_out_unbiased[1] += 1)
        else
            ok_y  = abp_add_weighted_value!(acc.counts_path_y_pos_biased, edges_path_y, y, weight)
            ok_xy = abp_add_weighted_joint!(acc.counts_path_xy_pos_biased, edges_path_x, edges_path_y, x, y, weight)
            ok_y  || (acc.n_path_y_pos_out_biased[1] += 1)
            ok_xy || (acc.n_path_xy_pos_out_biased[1] += 1)
        end
    end
end

function abp_maybe_save_path!(acc, sys, iw::Int, xT_now, yT_now, ymean_now, yint_now, bias_value_now, w_path;
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
