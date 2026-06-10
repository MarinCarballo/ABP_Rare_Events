using Test
using ABPNoiseSweep

@testset "configuration" begin
    cfg = ABPNoiseSweepConfig(
        D_values = [0.1, 0.01, 0.005],
        n_iter = 90,
        D_scaling_reference = 0.01,
        move_weights = [0.0, 0.05, 0.55, 0.40],
    )
    abp_validate_config!(cfg)
    @test abp_n_iter_for_D(cfg, 0.1) == 28
    @test abp_n_iter_for_D(cfg, 0.01) == 90
    @test abp_n_iter_for_D(cfg, 0.005) == 127
end

@testset "trajectory construction" begin
    abp = ABP(Dt=0.01, Dr=0.01, v=0.36)
    traj = ABPTrajectory(abp, 0.1; dt=0.01)
    @test length(traj.xs) == length(traj.θs)
    @test length(traj.ξs) == length(traj.xs) - 1
    @test isfinite(endpoint_x(traj))
end
