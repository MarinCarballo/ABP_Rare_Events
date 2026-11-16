using Random, StaticArrays, MonteCarloX

@testset "Work, crossing, and theta occupation conventions" begin
    sys = ABPTrajectory(Xoshiro(1), ABP(v=0.31), 0.03; dt=0.01)
    sys.xs .= [SVector(-1.0, 0.0), SVector(-0.1, 1.0),
               SVector(0.8, 2.0), SVector(1.1, 3.0)]
    # Cross the wrapping seam; work must use these continuous stored angles.
    sys.θs .= [2π - 0.1, 2π + 0.2, 2π + 0.4, 2π + 0.7]
    expected = sys.v * sum(
        cos((sys.θs[k] + sys.θs[k+1]) / 2) * (sys.xs[k+1][1] - sys.xs[k][1]) +
        sin((sys.θs[k] + sys.θs[k+1]) / 2) * (sys.xs[k+1][2] - sys.xs[k][2])
        for k in 1:3
    )
    metrics = active_work_and_first_lr_crossing(sys)
    @test metrics.active_work ≈ expected
    @test normalized_active_work(sys) ≈ expected / (sys.v^2 * sys.T)
    @test metrics.has_crossing
    @test metrics.crossing_y ≈ 1 + 1/9
    @test metrics.crossing_time ≈ (1 + 1/9) * sys.dt

    ex, ey, et = [-1.5, 0.0, 1.5], [-4.0, 0.0, 4.0], [0.0, π, 2π]
    new_acc() = ABPNoiseSweep.abp_make_prod_accumulator(ex, ey, ey, ey, ex, ey, et, 1, 1)
    acc, plain = new_acc(), new_acc()
    fused = ABPNoiseSweep.abp_add_whole_path_conditioned_windows!(
        acc, sys, [1], ex, ey, et, 1, 2.0;
        path_time_stride=2, path_x_min=-1.0, compute_route_metrics=true,
    )
    ABPNoiseSweep.abp_add_whole_path_conditioned_windows!(
        plain, sys, [1], ex, ey, et, 1, 2.0;
        path_time_stride=2, path_x_min=-1.0, compute_route_metrics=false,
    )
    @test fused == metrics # Work remains full resolution despite histogram thinning.
    @test acc.counts_path_xtheta_biased == plain.counts_path_xtheta_biased
    @test sum(acc.counts_path_xtheta_biased) == 1
    @test acc.counts_path_xtheta_biased[2, 1] == 1
    @test acc.counts_path_xtheta_unbiased[2, 1] == 2
    @test sum(ABPNoiseSweep.abp_pdf2_from_mass(acc.counts_path_xtheta_unbiased, ex, et) .* (diff(ex) .* diff(et)')) ≈ 1

    seam = new_acc()
    for θ in (-0.1, 0.1, 2π + 0.1, 4π - 0.1)
        ABPNoiseSweep.abp_add_path_point_conditioned_windows!(
            seam, SVector(0.7, 0.0), θ, [1], ex, ey, et, true, 3.0, true, -1.0,
        )
    end
    @test seam.counts_path_xtheta_biased[2, :] == [2, 2]
    @test seam.counts_path_xtheta_unbiased[2, :] == [6, 6]
end

@testset "Histogram boundaries and reweighting" begin
    edges = [0.0, 1.0, 3.0]
    @test ABPNoiseSweep.abp_bin_index_from_edges(0.0, edges) == 1
    @test ABPNoiseSweep.abp_bin_index_from_edges(1.0, edges) == 2
    @test ABPNoiseSweep.abp_bin_index_from_edges(3.0, edges) == 2
    @test ABPNoiseSweep.abp_bin_index_from_edges(-0.1, edges) === nothing
    @test ABPNoiseSweep.abp_bin_index_from_edges(NaN, edges) === nothing
    pdf = ABPNoiseSweep.abp_pdf_from_mass([2.0, 6.0], edges)
    @test sum(pdf .* diff(edges)) ≈ 1
    @test ABPNoiseSweep.abp_unbias_weight_from_value(0.5, edges, [2.0, 4.0], 2.0) == 1
    @test ABPNoiseSweep.abp_unbias_weight_from_value(1.5, edges, [2.0, 4.0], 2.0) ≈ exp(-2)
end

@testset "Preserved MonteCarloX diagnostic caveats" begin
    histogram = BinnedObject(-1.5:1.0:1.5, 0.0)
    @test flatness(histogram, -1.0, 1.0) == Inf
    histogram.values[1] = 7.0
    # This intentionally captures the dependency's occupied-bin definition.
    @test flatness(histogram, -1.0, 1.0; criterion=:max_over_mean) == 1
    @test flatness(histogram, -1.0, 1.0; criterion=:mean_over_min) == 1
    rt = Roundtrips(-1.0, 1.0)
    for x in (-1.0, 1.0, -1.0)
        MonteCarloX.update!(rt, x)
    end
    @test rt.count == 2 # two one-way traversals, one complete return cycle

    sys = ABPTrajectory(Xoshiro(3), ABP(), 0.02)
    alg = Multicanonical(Xoshiro(4), BinnedObject(-2.0:0.1:2.0, 0.0))
    ABPNoiseSweep.update!(sys, alg, endpoint_x, 1e6; bins=-2.0:0.1:2.0, potential_active=false)
    @test alg.steps == 1
    @test sum(ensemble(alg).histogram.values) == 0 # prior-rejected visits omitted
    ABPNoiseSweep.reflection_update!(sys, alg, endpoint_x; bins=-2.0:0.1:2.0)
    @test alg.steps == 3 # accept! plus the move's existing step increment
    @test sum(ensemble(alg).histogram.values) == 1
end

@testset "New configuration validation and environment controls" begin
    @test_throws ErrorException abp_validate_config!(ABPNoiseSweepConfig(integrator=:invalid))
    @test_throws ErrorException abp_validate_config!(ABPNoiseSweepConfig(noise_proposal=:invalid))
    @test_throws ErrorException abp_validate_config!(ABPNoiseSweepConfig(block_pcn_beta=1.1))
    withenv("ABP_INTEGRATOR"=>"crank_nicolson", "ABP_SAVE_MUCA_MOVIE_PATHS"=>"true",
            "ABP_SAVED_PATH_TIME_THIN"=>"1", "ABP_NOISE_PROPOSAL"=>"pcn") do
        cfg = abp_config_from_args(String[])
        @test cfg.integrator === :crank_nicolson
        @test cfg.noise_proposal === :pcn
        @test cfg.save_muca_movie_paths
        @test cfg.saved_path_time_thin == 1
        @test abp_config_from_args(["--integrator=euler"]).integrator === :euler
    end
end
