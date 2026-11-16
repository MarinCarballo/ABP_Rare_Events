using Test, Random, StaticArrays, LinearAlgebra, JLD2

# Literal pre-parameterization formulas: changing lambda to an option must not
# change the historical lambda=2.5 arithmetic or the Euler noise map.
function lambda_legacy_potential(position)
    x, y = position
    lambda_ = 2.5
    k = 0.3
    f_0 = 0.5 * lambda_ * y^2
    f_bar = 1.0 + 0.5 * k * y^2
    denom = f_0 + f_bar
    f_mix = denom == 0.0 ? 0.0 : (f_0 * f_bar) / denom
    return (x - 1.0)^2 * (x + 1.0)^2 * (f_bar - f_mix) + f_mix
end

function lambda_legacy_gradient(position)
    x, y = position
    lambda_ = 2.5
    k = 0.3
    gxy = (x - 1.0)^2 * (x + 1.0)^2
    dg_dx = 4.0 * x * (x^2 - 1.0)
    f_0 = 0.5 * lambda_ * y^2
    f_bar = 1.0 + 0.5 * k * y^2
    denom = f_0 + f_bar
    f_mix = denom == 0.0 ? 0.0 : (f_0 * f_bar) / denom
    dfbar_dy = k * y
    dfmix_dy = denom == 0.0 ? 0.0 : y * (lambda_ * f_bar^2 + k * f_0^2) / denom^2
    return SVector(dg_dx * (f_bar - f_mix), gxy * (dfbar_dy - dfmix_dy) + dfmix_dy)
end

function lambda_legacy_euler(sys)
    xs, angles = copy(sys.xs), copy(sys.θs)
    x, θ = xs[1], angles[1]
    for k in eachindex(sys.ξs)
        ξ = sys.ξs[k]
        grad = lambda_legacy_gradient(x)
        x = x + sys.v * SVector(cos(θ), sin(θ)) * sys.dt +
            sys.σ_T * SVector(ξ[1], ξ[2]) - grad * sys.dt
        θ = θ + sys.σ_R * ξ[3]
        xs[k + 1], angles[k + 1] = x, θ
    end
    return xs, angles
end

@testset "Potential lambda configuration, defaults, and historical arithmetic" begin
    @test ABP().potential_lambda == 2.5
    @test ABPNoiseSweepConfig().potential_lambda == 2.5
    @test ABP(potential_lambda=8).potential_lambda == 8.0
    for λ in (0.0, -1.0, NaN, Inf)
        @test_throws ErrorException ABP(potential_lambda=λ)
        @test_throws ErrorException abp_validate_config!(ABPNoiseSweepConfig(potential_lambda=λ))
    end
    withenv("ABP_POTENTIAL_LAMBDA" => "8") do
        @test abp_config_from_args(String[]).potential_lambda == 8.0
        @test abp_config_from_args(["--potential-lambda=2.5"]).potential_lambda == 2.5
    end
    sys = ABPTrajectory(Xoshiro(84), ABP(), 0.5; x0=[-1.0, 0.2], potential_active=true)
    @test sys.potential_lambda == 2.5
    for position in (SVector(-1.2, -0.7), SVector(0.2, 0.0), SVector(0.6, 1.1))
        sys.xs[1] = position
        @test reinterpret(UInt64, ABPNoiseSweep.potential(sys, 1)) ==
              reinterpret(UInt64, lambda_legacy_potential(position))
        actual, expected = ABPNoiseSweep.gradient(position), lambda_legacy_gradient(position)
        @test reinterpret(UInt64, actual[1]) == reinterpret(UInt64, expected[1])
        @test reinterpret(UInt64, actual[2]) == reinterpret(UInt64, expected[2])
    end
    # Reconstruct a fresh complete path with the original gradient expression.
    sys = ABPTrajectory(Xoshiro(84), ABP(), 0.5; x0=[-1.0, 0.2], potential_active=true)
    expected_xs, expected_angles = lambda_legacy_euler(sys)
    @test reinterpret(UInt64, sys.xs) == reinterpret(UInt64, expected_xs)
    @test reinterpret(UInt64, sys.θs) == reinterpret(UInt64, expected_angles)
end

@testset "Both lambda values parameterize potential, gradient, Hessian, and CN" begin
    δ = 1e-5
    for λ in (2.5, 8.0)
        abp = ABP(Dt=0.025, Dr=0.025, v=0.31, potential_lambda=λ)
        sys = ABPTrajectory(Xoshiro(91), abp, 0.1; potential_active=true)
        @test sys.potential_lambda == λ
        for position in (SVector(-1.2, -0.7), SVector(0.2, 0.0), SVector(0.6, 1.1))
            sys.xs[1] = position
            @test ABPNoiseSweep.gradient(sys, 1) == ABPNoiseSweep.gradient(position, λ)
            hessian = ABPNoiseSweep.potential_hessian(position, λ)
            for j in 1:2
                step = j == 1 ? SVector(δ, 0.0) : SVector(0.0, δ)
                sys.xs[1] = position + step
                u_plus = ABPNoiseSweep.potential(sys, 1)
                sys.xs[1] = position - step
                u_minus = ABPNoiseSweep.potential(sys, 1)
                @test ABPNoiseSweep.gradient(position, λ)[j] ≈ (u_plus - u_minus) / (2δ) atol=1e-8
                derivative = (ABPNoiseSweep.gradient(position + step, λ) -
                              ABPNoiseSweep.gradient(position - step, λ)) / (2δ)
                @test hessian[:, j] ≈ derivative atol=1e-7
            end
        end
        cn = ABPTrajectory(Xoshiro(91), abp, 0.2;
                           x0=[-1.0, 0.2], potential_active=true, integrator=:crank_nicolson)
        for k in eachindex(cn.ξs)
            x, x_next = cn.xs[k], cn.xs[k + 1]
            θ, θ_next = cn.θs[k], cn.θs[k + 1]
            ξ = cn.ξs[k]
            expected = cn.v * cn.dt / 2 * (SVector(cos(θ), sin(θ)) + SVector(cos(θ_next), sin(θ_next))) +
                       cn.σ_T * SVector(ξ[1], ξ[2]) - cn.dt / 2 *
                       (ABPNoiseSweep.gradient(x, λ) + ABPNoiseSweep.gradient(x_next, λ))
            @test norm(x_next - x - expected, Inf) <= 5e-12
        end
    end
    low = ABPTrajectory(Xoshiro(84), ABP(potential_lambda=2.5), 0.2;
                        x0=[-1.0, 0.2], potential_active=true)
    high = ABPTrajectory(Xoshiro(84), ABP(potential_lambda=8.0), 0.2;
                         x0=[-1.0, 0.2], potential_active=true)
    @test low.ξs == high.ξs
    @test low.θs == high.θs
    @test low.xs != high.xs
    @test rand(low.rng, 8) == rand(high.rng, 8)
    free_low = ABPTrajectory(Xoshiro(84), ABP(potential_lambda=2.5), 0.2)
    free_high = ABPTrajectory(Xoshiro(84), ABP(potential_lambda=8.0), 0.2)
    @test free_low.xs == free_high.xs
end

@testset "Lambda eight MUCA/production/output smoke" begin
    mktempdir() do directory
        cfg = ABPNoiseSweepConfig(
            potential_lambda=8.0, trajectory_T=0.05, dt=0.01, D_values=[0.025], v=0.31,
            scale_n_iter_with_D=false, n_iter=1, n_iter_steps_per_iter=8, n_therm_muca=0,
            n_prod_chains=1, n_prod_obs_total=8, n_therm_prod=0, prod_stride=2,
            path_observation_stride=1, active_work_observation_stride=2,
            max_saved_paths_per_window=1, n_y_bins=2, n_y_int_bins=2,
            n_path_x_bins=2, n_path_theta_bins=2, output_dir=directory,
        )
        abp_validate_config!(cfg)
        muca = ABPNoiseSweep.abp_run_muca_one_case(cfg, 0.025)
        @test muca.abp.potential_lambda == 8.0
        prod = ABPNoiseSweep.abp_run_production_one_case(cfg, 0.025, muca)
        @test sum(prod.acc.counts_x_T_biased) == 8
        filename = ABPNoiseSweep.abp_save_case_jld2(cfg, 0.025, muca, prod)
        jldopen(filename, "r") do file
            @test file["metadata/potential_lambda"] == 8.0
            @test file["metadata/potential_k"] == 0.3
        end
        files = ABPNoiseSweep.abp_export_case_data_csvs(filename; output_dir=joinpath(directory, "data"))
        @test !isempty(files)
        cfg.potential_lambda = 2.5
        @test_throws ErrorException ABPNoiseSweep.abp_run_production_one_case(cfg, 0.025, muca)
    end
end
