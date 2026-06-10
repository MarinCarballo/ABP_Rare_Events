# JLD2 and CSV outputs for downstream plotting.

function abp_save_case_jld2(cfg::ABPNoiseSweepConfig, D::Real, muca, prod)
    abp_ensure_dir(cfg.output_dir)
    tag = abp_run_tag(D)
    file_path = joinpath(cfg.output_dir, "abp_endpoint_conditioned_$(tag).jld2")

    acc = prod.acc
    endpoint_window_specs = prod.endpoint_window_specs
    window_keys = prod.window_keys

    jldopen(file_path, "w") do file
        # Metadata
        file["metadata/D"] = Float64(D)
        file["metadata/Dt"] = Float64(D)
        file["metadata/Dr"] = Float64(D)
        file["metadata/v"] = cfg.v
        file["metadata/move_weights"] = cfg.move_weights
        file["metadata/move_weights_order"] = ["reflection_update", "theta0_update", "block_update", "single_noise_update"]
        file["metadata/trajectory_T"] = cfg.trajectory_T
        file["metadata/dt"] = cfg.dt
        file["metadata/x0"] = cfg.x0_vec
        file["metadata/potential_active"] = cfg.potential_active
        file["metadata/path_time_stride"] = cfg.path_time_stride
        file["metadata/saved_path_time_thin"] = cfg.saved_path_time_thin

        file["metadata/bias_observable"] = "endpoint_x"
        file["metadata/bias_min"] = cfg.bias_min
        file["metadata/bias_max"] = cfg.bias_max
        file["metadata/dbias"] = cfg.dbias
        file["metadata/xT_min"] = cfg.xT_min
        file["metadata/xT_max"] = cfg.xT_max
        file["metadata/xT_min_extend"] = muca.xT_min_extend
        file["metadata/xT_max_extend"] = muca.xT_max_extend
        file["metadata/tail_temperature"] = muca.tail_temperature
        file["metadata/tail_slope"] = -1 / muca.tail_temperature
        file["metadata/D_scaling_reference"] = muca.D_scaling_reference
        file["metadata/scale_n_iter_with_D"] = cfg.scale_n_iter_with_D
        file["metadata/n_iter_factor"] = muca.n_iter_factor
        file["metadata/n_iter_effective"] = muca.n_iter_effective
        file["metadata/block_dxi"] = muca.block_dxi
        file["metadata/local_dxi"] = muca.local_dxi

        file["metadata/n_iter"] = cfg.n_iter
        file["metadata/n_iter_steps_per_iter"] = cfg.n_iter_steps_per_iter
        file["metadata/n_therm_muca"] = cfg.n_therm_muca
        file["metadata/flatness_threshold"] = cfg.flatness_threshold
        file["metadata/roundtrip_target"] = cfg.roundtrip_target
        file["metadata/roundtrip_convergence_window"] = cfg.roundtrip_convergence_window
        file["metadata/roundtrip_convergence_rtol"] = cfg.roundtrip_convergence_rtol
        file["metadata/n_muca_chains"] = muca.n_chains
        file["metadata/n_muca_total_sweeps_approx"] = muca.n_muca_total
        file["metadata/t_muca_seconds"] = muca.t_muca

        file["metadata/n_prod_obs_total"] = cfg.n_prod_obs_total
        file["metadata/n_prod_chains"] = prod.n_prod_chains
        file["metadata/n_prod_steps_by_chain"] = prod.n_steps_chain
        file["metadata/n_therm_prod_per_chain"] = cfg.n_therm_prod
        file["metadata/t_prod_seconds"] = prod.t_prod
        file["metadata/prod_stride"] = cfg.prod_stride
        file["metadata/roundtrip_stride"] = cfg.roundtrip_stride

        # MUCA data
        file["muca/bins_bias"] = muca.edges_bias
        file["muca/centers_bias"] = muca.centers_bias
        file["muca/iter_hist_values"] = hcat(muca.iter_hist_values...)
        file["muca/iter_logw_values"] = hcat(muca.iter_logw_values...)
        file["muca/last_hist_values"] = muca.iter_hist_values[end]
        file["muca/last_logw_values"] = muca.iter_logw_values[end]
        file["muca/iter_accept"] = muca.iter_accept
        file["muca/iter_flatness_maxmean"] = muca.iter_flatness_maxmean
        file["muca/iter_flatness_meanmin"] = muca.iter_flatness_meanmin
        file["muca/iter_roundtrips"] = muca.iter_roundtrips
        file["muca/iter_sampling_steps"] = muca.iter_sampling_steps
        file["muca/iter_total_mcmc_steps"] = muca.iter_total_mcmc_steps
        file["muca/iter_roundtrips_per_sampling_step"] = muca.iter_roundtrips_per_sampling_step
        file["muca/iter_steps_per_target_roundtrips"] = muca.iter_steps_per_target_roundtrips
        file["muca/roundtrip_target"] = cfg.roundtrip_target
        file["muca/roundtrip_steps_estimate"] = muca.roundtrip_steps_summary.steps_estimate
        file["muca/roundtrip_steps_converged"] = muca.roundtrip_steps_summary.converged
        file["muca/roundtrip_steps_rel_half_range"] = muca.roundtrip_steps_summary.rel_half_range
        file["muca/roundtrip_steps_window_n"] = muca.roundtrip_steps_summary.window_n
        file["muca/roundtrip_steps_note"] = muca.roundtrip_steps_summary.note
        file["muca/logw_shift"] = prod.logw_shift

        # Histogram edges
        file["histograms/bins_x_T"] = prod.edges_x_T
        file["histograms/bins_y_T"] = prod.edges_y_T
        file["histograms/bins_y_mean"] = prod.edges_y_mean
        file["histograms/bins_y_int"] = prod.edges_y_int
        file["histograms/bins_path_x"] = prod.edges_path_x
        file["histograms/bins_path_y"] = prod.edges_path_y

        # Endpoint-x histograms
        file["histograms/biased/x_T"] = acc.counts_x_T_biased
        file["histograms/unbiased/x_T"] = acc.counts_x_T_unbiased

        # Requested whole-trajectory endpoint-positive branch.
        file["endpoint_positive_condition/condition"] = "x(T) > 0"
        file["endpoint_positive_condition/biased/path_y"] = acc.counts_path_y_pos_biased
        file["endpoint_positive_condition/unbiased/path_y"] = acc.counts_path_y_pos_unbiased
        file["endpoint_positive_condition/biased/path_x_y"] = acc.counts_path_xy_pos_biased
        file["endpoint_positive_condition/unbiased/path_x_y"] = acc.counts_path_xy_pos_unbiased
        file["endpoint_positive_condition/diagnostics/out_path_y_biased"] = acc.n_path_y_pos_out_biased[1]
        file["endpoint_positive_condition/diagnostics/out_path_y_unbiased"] = acc.n_path_y_pos_out_unbiased[1]
        file["endpoint_positive_condition/diagnostics/out_path_x_y_biased"] = acc.n_path_xy_pos_out_biased[1]
        file["endpoint_positive_condition/diagnostics/out_path_x_y_unbiased"] = acc.n_path_xy_pos_out_unbiased[1]

        # Reweighting diagnostics
        file["histograms/reweighting/logw_shift"] = prod.logw_shift
        file["histograms/reweighting/sum_w"] = acc.sum_w_all[1]
        file["histograms/reweighting/sum_w2"] = acc.sum_w2_all[1]
        file["histograms/reweighting/ess"] = acc.sum_w2_all[1] > 0.0 ? acc.sum_w_all[1]^2 / acc.sum_w2_all[1] : NaN
        file["histograms/reweighting/n_reweighted_samples"] = acc.n_rew_all[1]

        # Production roundtrips
        file["roundtrips/observable"] = "endpoint_x"
        file["roundtrips/bounds"] = [prod.production_rt_min, prod.production_rt_max]
        file["roundtrips/stride"] = cfg.roundtrip_stride
        file["roundtrips/steps"] = acc.rt_steps
        file["roundtrips/counts"] = acc.rt_counts
        file["roundtrips/bias_values"] = acc.rt_values
        file["roundtrips/final_count"] = acc.rt_final_count[1]

        # Conditional endpoint windows
        file["conditional_windows/order"] = window_keys
        for (iw, (key_sym, lo, hi, label)) in enumerate(endpoint_window_specs)
            key = string(key_sym)
            file["conditional_windows/$key/label"] = label
            file["conditional_windows/$key/x_min"] = lo
            file["conditional_windows/$key/x_max"] = hi

            file["conditional_windows/$key/biased/y_T"]       = acc.counts_y_T_biased[iw]
            file["conditional_windows/$key/biased/y_mean"]    = acc.counts_y_mean_biased[iw]
            file["conditional_windows/$key/biased/y_int"]     = acc.counts_y_int_biased[iw]
            file["conditional_windows/$key/biased/x_T_y_T"]   = acc.counts_xy_T_biased[iw]

            file["conditional_windows/$key/unbiased/y_T"]     = acc.counts_y_T_unbiased[iw]
            file["conditional_windows/$key/unbiased/y_mean"]  = acc.counts_y_mean_unbiased[iw]
            file["conditional_windows/$key/unbiased/y_int"]   = acc.counts_y_int_unbiased[iw]
            file["conditional_windows/$key/unbiased/x_T_y_T"] = acc.counts_xy_T_unbiased[iw]

            file["conditional_windows/$key/diagnostics/n_biased"]     = acc.n_biased_by_window[iw]
            file["conditional_windows/$key/diagnostics/n_reweighted"] = acc.n_rew_by_window[iw]
            file["conditional_windows/$key/diagnostics/sum_w"]        = acc.sum_w_by_window[iw]
            file["conditional_windows/$key/diagnostics/sum_w2"]       = acc.sum_w2_by_window[iw]
            file["conditional_windows/$key/diagnostics/ess"]          = acc.sum_w2_by_window[iw] > 0.0 ? acc.sum_w_by_window[iw]^2 / acc.sum_w2_by_window[iw] : NaN
            file["conditional_windows/$key/diagnostics/out_y_T"]      = acc.n_y_T_out[iw]
            file["conditional_windows/$key/diagnostics/out_y_mean"]   = acc.n_y_mean_out[iw]
            file["conditional_windows/$key/diagnostics/out_y_int"]    = acc.n_y_int_out[iw]
            file["conditional_windows/$key/diagnostics/out_x_T_y_T"]  = acc.n_xy_T_out[iw]

            file["saved_trajectories/$key/n_saved"] = length(acc.saved_paths[iw])
            for (j, path) in enumerate(acc.saved_paths[iw])
                file["saved_trajectories/$key/$j/xs"] = path.xs
                file["saved_trajectories/$key/$j/theta0"] = path.theta0
                file["saved_trajectories/$key/$j/endpoint_x"] = path.endpoint_x
                file["saved_trajectories/$key/$j/endpoint_y"] = path.endpoint_y
                file["saved_trajectories/$key/$j/y_mean"] = path.y_mean
                file["saved_trajectories/$key/$j/y_int"] = path.y_int
                file["saved_trajectories/$key/$j/bias_value"] = path.bias_value
                file["saved_trajectories/$key/$j/unbias_weight_shifted"] = path.unbias_weight_shifted
            end
        end
    end

    println("Saved case file: ", file_path)
    return file_path
end

# -----------------------------
# CSV/data export helpers
# -----------------------------

function abp_csv_field(x)
    if x === nothing
        return ""
    elseif x isa AbstractString
        s = String(x)
        if occursin(",", s) || occursin("\"", s) || occursin("\n", s) || occursin("\r", s)
            return "\"" * replace(s, "\"" => "\"\"") * "\""
        else
            return s
        end
    elseif x isa Bool
        return x ? "true" : "false"
    else
        return string(x)
    end
end

function abp_write_csv(path::AbstractString, header::Vector{String}, rows)
    abp_ensure_dir(dirname(path))
    open(path, "w") do io
        println(io, join(abp_csv_field.(header), ","))
        for row in rows
            println(io, join(abp_csv_field.(collect(row)), ","))
        end
    end
    return path
end

function abp_write_key_value_csv(path::AbstractString, pairs)
    return abp_write_csv(path, ["key", "value"], ((first(pair), last(pair)) for pair in pairs))
end

function abp_safe_pdf_from_mass(counts::AbstractVector, edges::AbstractVector)
    return abp_pdf_from_mass(Float64.(counts), Float64.(edges))
end

function abp_safe_pdf2_from_mass(counts::AbstractMatrix, x_edges::AbstractVector, y_edges::AbstractVector)
    return abp_pdf2_from_mass(Float64.(counts), Float64.(x_edges), Float64.(y_edges))
end

function abp_export_case_data_csvs(file_path::AbstractString; output_dir::AbstractString=joinpath(dirname(file_path), "data"))
    abp_ensure_dir(output_dir)
    tag = replace(basename(file_path), ".jld2" => "")
    case_dir = abp_ensure_dir(joinpath(output_dir, tag))
    outs = String[]

    jldopen(file_path, "r") do file
        D = Float64(file["metadata/D"])
        x0 = Float64(file["metadata/x0"][1])

        centers_bias = collect(file["muca/centers_bias"])
        iter_hist = Array(file["muca/iter_hist_values"])
        iter_logw = Array(file["muca/iter_logw_values"])
        last_hist = collect(file["muca/last_hist_values"])
        last_logw = collect(file["muca/last_logw_values"])
        i0 = argmin(abs.(centers_bias .- x0))
        inv_weight_rel_x0 = exp.(-(last_logw .- last_logw[i0]))

        push!(outs, abp_write_csv(
            joinpath(case_dir, "muca_last_histogram_logweights.csv"),
            ["x_T_center", "last_hist_count", "last_logweight", "inverse_weight_relative_to_x0"],
            zip(centers_bias, last_hist, last_logw, inv_weight_rel_x0),
        ))

        n_iter = size(iter_hist, 2)
        if n_iter > 0
            push!(outs, abp_write_csv(
                joinpath(case_dir, "muca_iteration_diagnostics.csv"),
                ["iteration", "acceptance", "flatness_max_over_mean", "flatness_mean_over_min", "roundtrips", "sampling_steps", "total_mcmc_steps", "roundtrips_per_sampling_step", "steps_per_target_roundtrips"],
                ((i,
                  file["muca/iter_accept"][i],
                  file["muca/iter_flatness_maxmean"][i],
                  file["muca/iter_flatness_meanmin"][i],
                  file["muca/iter_roundtrips"][i],
                  file["muca/iter_sampling_steps"][i],
                  file["muca/iter_total_mcmc_steps"][i],
                  file["muca/iter_roundtrips_per_sampling_step"][i],
                  file["muca/iter_steps_per_target_roundtrips"][i]) for i in 1:n_iter),
            ))
        end

        x_edges = collect(file["histograms/bins_x_T"])
        x_centers = abp_centers_from_edges(x_edges)
        x_b = collect(file["histograms/biased/x_T"])
        x_u = collect(file["histograms/unbiased/x_T"])
        x_b_pdf = abp_safe_pdf_from_mass(x_b, x_edges)
        x_u_pdf = abp_safe_pdf_from_mass(x_u, x_edges)
        push!(outs, abp_write_csv(
            joinpath(case_dir, "hist_endpoint_x_T.csv"),
            ["x_T_center", "biased_count", "unbiased_count", "biased_pdf", "unbiased_pdf"],
            zip(x_centers, x_b, x_u, x_b_pdf, x_u_pdf),
        ))

        path_y_edges = collect(file["histograms/bins_path_y"])
        path_y_centers = abp_centers_from_edges(path_y_edges)
        path_y_b = collect(file["endpoint_positive_condition/biased/path_y"])
        path_y_u = collect(file["endpoint_positive_condition/unbiased/path_y"])
        path_y_b_pdf = abp_safe_pdf_from_mass(path_y_b, path_y_edges)
        path_y_u_pdf = abp_safe_pdf_from_mass(path_y_u, path_y_edges)
        push!(outs, abp_write_csv(
            joinpath(case_dir, "endpoint_positive_path_y.csv"),
            ["path_y_center", "biased_count", "unbiased_count", "biased_pdf", "unbiased_pdf"],
            zip(path_y_centers, path_y_b, path_y_u, path_y_b_pdf, path_y_u_pdf),
        ))

        path_x_edges = collect(file["histograms/bins_path_x"])
        path_x_centers = abp_centers_from_edges(path_x_edges)
        path_xy_b = Array(file["endpoint_positive_condition/biased/path_x_y"])
        path_xy_u = Array(file["endpoint_positive_condition/unbiased/path_x_y"])
        path_xy_b_pdf = abp_safe_pdf2_from_mass(path_xy_b, path_x_edges, path_y_edges)
        path_xy_u_pdf = abp_safe_pdf2_from_mass(path_xy_u, path_x_edges, path_y_edges)
        push!(outs, abp_write_csv(
            joinpath(case_dir, "endpoint_positive_path_x_y_long.csv"),
            ["path_x_center", "path_y_center", "biased_count", "unbiased_count", "biased_pdf", "unbiased_pdf"],
            ((path_x_centers[ix], path_y_centers[iy], path_xy_b[ix, iy], path_xy_u[ix, iy], path_xy_b_pdf[ix, iy], path_xy_u_pdf[ix, iy])
             for ix in eachindex(path_x_centers) for iy in eachindex(path_y_centers)),
        ))

        keys_order = collect(file["conditional_windows/order"])
        y_edges = collect(file["histograms/bins_y_T"])
        mean_edges = collect(file["histograms/bins_y_mean"])
        int_edges = collect(file["histograms/bins_y_int"])
        y_centers = abp_centers_from_edges(y_edges)
        mean_centers = abp_centers_from_edges(mean_edges)
        int_centers = abp_centers_from_edges(int_edges)

        summary_pairs = Pair{String, Any}[
            "source_file" => basename(file_path),
            "D" => D,
            "trajectory_T" => file["metadata/trajectory_T"],
            "dt" => file["metadata/dt"],
            "v" => file["metadata/v"],
            "potential_active" => file["metadata/potential_active"],
            "n_iter" => file["metadata/n_iter"],
            "n_muca_chains" => file["metadata/n_muca_chains"],
            "n_prod_obs_total" => file["metadata/n_prod_obs_total"],
            "n_prod_chains" => file["metadata/n_prod_chains"],
            "roundtrip_target" => file["metadata/roundtrip_target"],
            "D_scaling_reference" => file["metadata/D_scaling_reference"],
            "scale_n_iter_with_D" => file["metadata/scale_n_iter_with_D"],
            "n_iter_factor" => file["metadata/n_iter_factor"],
            "n_iter_effective" => file["metadata/n_iter_effective"],
            "block_dxi" => file["metadata/block_dxi"],
            "local_dxi" => file["metadata/local_dxi"],
            "roundtrip_steps_estimate" => file["muca/roundtrip_steps_estimate"],
            "roundtrip_steps_converged" => file["muca/roundtrip_steps_converged"],
            "roundtrip_steps_rel_half_range" => file["muca/roundtrip_steps_rel_half_range"],
            "reweighting_sum_w" => file["histograms/reweighting/sum_w"],
            "reweighting_sum_w2" => file["histograms/reweighting/sum_w2"],
            "reweighting_ess" => file["histograms/reweighting/ess"],
            "reweighting_n_samples" => file["histograms/reweighting/n_reweighted_samples"],
        ]

        for key in keys_order
            label = file["conditional_windows/$key/label"]
            push!(summary_pairs, "window_$(key)_label" => label)
            push!(summary_pairs, "window_$(key)_n_biased" => file["conditional_windows/$key/diagnostics/n_biased"])
            push!(summary_pairs, "window_$(key)_n_reweighted" => file["conditional_windows/$key/diagnostics/n_reweighted"])
            push!(summary_pairs, "window_$(key)_ess" => file["conditional_windows/$key/diagnostics/ess"])

            rows = Iterators.flatten((
                (("y_T", y_centers[i],
                  file["conditional_windows/$key/biased/y_T"][i],
                  file["conditional_windows/$key/unbiased/y_T"][i],
                  abp_safe_pdf_from_mass(collect(file["conditional_windows/$key/biased/y_T"]), y_edges)[i],
                  abp_safe_pdf_from_mass(collect(file["conditional_windows/$key/unbiased/y_T"]), y_edges)[i]) for i in eachindex(y_centers)),
                (("y_mean", mean_centers[i],
                  file["conditional_windows/$key/biased/y_mean"][i],
                  file["conditional_windows/$key/unbiased/y_mean"][i],
                  abp_safe_pdf_from_mass(collect(file["conditional_windows/$key/biased/y_mean"]), mean_edges)[i],
                  abp_safe_pdf_from_mass(collect(file["conditional_windows/$key/unbiased/y_mean"]), mean_edges)[i]) for i in eachindex(mean_centers)),
                (("y_int", int_centers[i],
                  file["conditional_windows/$key/biased/y_int"][i],
                  file["conditional_windows/$key/unbiased/y_int"][i],
                  abp_safe_pdf_from_mass(collect(file["conditional_windows/$key/biased/y_int"]), int_edges)[i],
                  abp_safe_pdf_from_mass(collect(file["conditional_windows/$key/unbiased/y_int"]), int_edges)[i]) for i in eachindex(int_centers)),
            ))
            push!(outs, abp_write_csv(
                joinpath(case_dir, "conditional_$(key)_one_dimensional_histograms.csv"),
                ["observable", "center", "biased_count", "unbiased_count", "biased_pdf", "unbiased_pdf"],
                rows,
            ))
        end

        push!(outs, abp_write_key_value_csv(joinpath(case_dir, "case_summary.csv"), summary_pairs))
    end

    println("Exported data CSV files for ", basename(file_path), " to ", case_dir)
    return outs
end

# -----------------------------
# Cross-noise roundtrip scaling diagnostic
# -----------------------------

function abp_read_roundtrip_scaling_rows(file_paths::Vector{String})
    rows = NamedTuple[]
    for file_path in file_paths
        jldopen(file_path, "r") do file
            D = Float64(file["metadata/D"])
            target = haskey(file, "muca/roundtrip_target") ? Int(file["muca/roundtrip_target"]) : Int(file["metadata/roundtrip_target"])
            steps = Float64(file["muca/roundtrip_steps_estimate"])
            converged = Bool(file["muca/roundtrip_steps_converged"])
            rel_half_range = Float64(file["muca/roundtrip_steps_rel_half_range"])
            final_rt = Int(collect(file["muca/iter_roundtrips"])[end])
            D_scaling_reference = haskey(file, "metadata/D_scaling_reference") ? Float64(file["metadata/D_scaling_reference"]) : NaN
            n_iter_factor = haskey(file, "metadata/n_iter_factor") ? Float64(file["metadata/n_iter_factor"]) : NaN
            n_iter_effective = haskey(file, "metadata/n_iter_effective") ? Int(file["metadata/n_iter_effective"]) : Int(file["metadata/n_iter"])
            block_dxi = haskey(file, "metadata/block_dxi") ? Float64(file["metadata/block_dxi"]) : NaN
            local_dxi = haskey(file, "metadata/local_dxi") ? Float64(file["metadata/local_dxi"]) : NaN
            push!(rows, (
                file_path = file_path,
                D = D,
                invD = 1.0 / D,
                roundtrip_target = target,
                steps_estimate = steps,
                converged = converged,
                rel_half_range = rel_half_range,
                final_iter_roundtrips = final_rt,
                D_scaling_reference = D_scaling_reference,
                n_iter_factor = n_iter_factor,
                n_iter_effective = n_iter_effective,
                block_dxi = block_dxi,
                local_dxi = local_dxi,
            ))
        end
    end
    return rows
end

function abp_power_law_fit(invD::AbstractVector{<:Real}, steps::AbstractVector{<:Real})
    mask = [isfinite(invD[i]) && isfinite(steps[i]) && invD[i] > 0 && steps[i] > 0 for i in eachindex(invD)]
    x = log.(Float64.(invD[mask]))
    y = log.(Float64.(steps[mask]))

    if length(x) < 2
        return (C = NaN, alpha = NaN, r2 = NaN, n = length(x), valid = false)
    end

    X = hcat(ones(length(x)), x)
    β = X \ y
    yhat = X * β
    ss_res = sum((y .- yhat) .^ 2)
    ss_tot = sum((y .- mean(y)) .^ 2)
    r2 = ss_tot > 0 ? 1 - ss_res / ss_tot : NaN

    return (C = exp(β[1]), alpha = β[2], r2 = r2, n = length(x), valid = true)
end

function abp_write_roundtrip_scaling_csv(rows, fit, output_dir::AbstractString)
    rows_csv = joinpath(output_dir, "roundtrip_scaling_rows.csv")
    open(rows_csv, "w") do io
        println(io, "file,D,invD,roundtrip_target,steps_estimate,converged,rel_half_range,final_iter_roundtrips,D_scaling_reference,n_iter_factor,n_iter_effective,block_dxi,local_dxi")
        for r in rows
            println(io, join((basename(r.file_path), r.D, r.invD, r.roundtrip_target,
                              r.steps_estimate, r.converged, r.rel_half_range, r.final_iter_roundtrips,
                              r.D_scaling_reference, r.n_iter_factor, r.n_iter_effective,
                              r.block_dxi, r.local_dxi), ","))
        end
    end

    fit_csv = joinpath(output_dir, "roundtrip_scaling_power_law_fit.csv")
    open(fit_csv, "w") do io
        println(io, "C,alpha,r2,n,valid")
        println(io, join((fit.C, fit.alpha, fit.r2, fit.n, fit.valid), ","))
    end

    return rows_csv, fit_csv
end

function abp_write_roundtrip_scaling_data(file_paths::Vector{String}; output_dir::AbstractString=".")
    abp_ensure_dir(output_dir)
    rows = abp_read_roundtrip_scaling_rows(file_paths)
    isempty(rows) && return String[]

    subset = [r for r in rows if isfinite(r.steps_estimate) && r.steps_estimate > 0]
    fit = abp_power_law_fit([r.invD for r in subset], [r.steps_estimate for r in subset])
    rows_csv, fit_csv = abp_write_roundtrip_scaling_csv(rows, fit, output_dir)

    println("Roundtrip scaling data saved:")
    println("  ", rows_csv)
    println("  ", fit_csv)
    println("  steps ≈ ", round(fit.C; sigdigits=4), " * (1/D)^", round(fit.alpha; sigdigits=4),
            " | R²=", round(fit.r2; digits=4), " | n=", fit.n)

    return [rows_csv, fit_csv]
end

# -----------------------------
# Main sweep driver
# -----------------------------
