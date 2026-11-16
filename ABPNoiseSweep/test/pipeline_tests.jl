using JLD2

@testset "Tiny threaded MUCA, production, and output pipeline" begin
    for (integrator, proposal) in ((:euler, :random_walk), (:crank_nicolson, :random_walk), (:euler, :pcn))
        mktempdir() do directory
            cfg = ABPNoiseSweepConfig(
                trajectory_T=0.1, dt=0.01, integrator=integrator, noise_proposal=proposal,
                D_values=[0.025], v=0.31, scale_n_iter_with_D=false,
                n_iter=2, n_iter_steps_per_iter=48, n_therm_muca=2,
                n_prod_chains=2, n_prod_obs_total=24, n_therm_prod=2,
                prod_stride=4, roundtrip_stride=4,
                roundtrip_avg_target_fraction=1e6,
                path_observation_stride=2, active_work_observation_stride=4,
                save_muca_movie_paths=true, muca_movie_stride=2, muca_movie_max_trajectories=5,
                save_movie_paths=true, movie_stride=2, movie_max_trajectories=5,
                movie_path_time_thin=3, saved_path_time_thin=1, max_saved_paths_per_window=2,
                n_y_bins=8, n_y_int_bins=8, n_path_x_bins=8, n_path_theta_bins=8,
                output_dir=directory,
            )
            abp_validate_config!(cfg)
            muca = ABPNoiseSweep.abp_run_muca_one_case(cfg, 0.025)
            prod = ABPNoiseSweep.abp_run_production_one_case(cfg, 0.025, muca)
            @test 1 <= length(muca.movie_paths) <= 5
            @test 1 <= length(prod.acc.movie_paths) <= 5
            @test all(p -> p.physical_times[end] ≈ 0.1, muca.movie_paths)
            @test issorted(getproperty.(muca.movie_paths, :sampling_step))
            filename = ABPNoiseSweep.abp_save_case_jld2(cfg, 0.025, muca, prod)
            exports = ABPNoiseSweep.abp_export_case_data_csvs(filename; output_dir=joinpath(directory, "data"))
            @test isfile(filename)
            @test any(p -> endswith(p, "muca_movie_trajectories_long.csv"), exports)
            @test any(p -> endswith(p, "movie_trajectories_long.csv"), exports)
            @test any(p -> endswith(p, "active_work_components_vs_crossing_y.csv"), exports)
            @test any(p -> endswith(p, "chain_active_work_components_vs_crossing_y.csv"), exports)
            jldopen(filename, "r") do file
                @test file["metadata/integrator"] == string(integrator)
                @test file["metadata/noise_proposal_selection"] == string(proposal)
                @test length(file["saved_trajectories/all/1/x"]) == 11
                @test file["saved_trajectories/all/1/physical_times"][end] ≈ 0.1
                for key in ("sum_Wx", "sum_Wy", "sum_Wx2", "sum_Wy2", "sum_WxWy")
                    @test haskey(file, "active_work/unbiased/$key")
                    @test haskey(file, "chain_statistics/chains/1/active_work/unbiased/$key")
                end
            end
            if integrator === :euler && proposal === :random_walk
                # Optional recording must not alter any proposals or histogram observations.
                cfg.save_movie_paths = cfg.save_muca_movie_paths = false
                baseline = ABPNoiseSweep.abp_run_muca_one_case(cfg, 0.025)
                baseprod = ABPNoiseSweep.abp_run_production_one_case(cfg, 0.025, baseline)
                @test baseline.iter_hist_values == muca.iter_hist_values
                @test baseline.iter_logw_values == muca.iter_logw_values
                @test baseprod.acc.counts_x_T_biased == prod.acc.counts_x_T_biased
                @test baseprod.acc.counts_x_T_unbiased == prod.acc.counts_x_T_unbiased
                @test baseprod.acc.counts_path_xy_unbiased == prod.acc.counts_path_xy_unbiased
                @test baseprod.acc.counts_path_xtheta_unbiased == prod.acc.counts_path_xtheta_unbiased
                cfg.save_movie_paths = true
                cfg.movie_max_trajectories = 0
                cfg.movie_chain_id = 999 # Disabled storage must not index this chain.
                disabled = ABPNoiseSweep.abp_run_production_one_case(cfg, 0.025, baseline)
                @test isempty(disabled.acc.movie_paths)
                @test disabled.acc.counts_x_T_unbiased == baseprod.acc.counts_x_T_unbiased
            end
        end
    end
end
