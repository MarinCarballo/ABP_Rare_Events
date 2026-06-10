# Top-level parameter sweep driver.

function run_abp_noise_sweep(cfg::ABPNoiseSweepConfig=ABPNoiseSweepConfig())
    abp_ensure_dir(cfg.output_dir)
    data_dir = abp_ensure_dir(joinpath(cfg.output_dir, "data"))

    println("Running ABP noise sweep.")
    println("trajectory_T = ", cfg.trajectory_T, " (physical integration time)")
    println("D values = ", cfg.D_values)
    println("move weights [reflection, theta0, block, local] = ", cfg.move_weights)
    println("Output dir = ", cfg.output_dir)

    result_files = String[]
    data_files = String[]

    for D in cfg.D_values
        muca = abp_run_muca_one_case(cfg, D)
        prod = abp_run_production_one_case(cfg, D, muca)
        file_path = abp_save_case_jld2(cfg, D, muca, prod)
        push!(result_files, file_path)

        if cfg.save_csv
            append!(data_files, abp_export_case_data_csvs(file_path; output_dir=data_dir))
        end
    end

    if cfg.save_csv
        append!(data_files, abp_write_roundtrip_scaling_data(result_files; output_dir=data_dir))
    end

    if cfg.save_plots || cfg.show_plots
        @warn "Plot generation is disabled in this data-only script. Use the JLD2/CSV outputs with your plotting code."
    end

    println("Done sweep.")
    println("JLD2 result files:")
    for f in result_files
        println("  ", f)
    end
    if cfg.save_csv
        println("CSV/data export files:")
        for f in data_files
            println("  ", f)
        end
    end

    return result_files, data_files
end
