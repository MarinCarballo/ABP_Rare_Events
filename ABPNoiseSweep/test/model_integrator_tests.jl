using Random
using StaticArrays
using LinearAlgebra
using MonteCarloX

# Literal Euler reference: catches changes to arithmetic, noise draws, or the
# initial-orientation convention when selecting the existing/default method.
function reference_euler_positions(sys; potential_active)
    xs, angles = copy(sys.xs), copy(sys.θs)
    for i in eachindex(sys.ξs)
        ξ, x, θ = sys.ξs[i], xs[i], angles[i]
        grad = potential_active ? ABPNoiseSweep.gradient(x) : SVector(0.0, 0.0)
        if potential_active
            xs[i + 1] = x + sys.v * SVector(cos(θ), sin(θ)) * sys.dt +
                        sys.σ_T * SVector(ξ[1], ξ[2]) - grad * sys.dt
        else
            xs[i + 1] = x + sys.v * SVector(cos(θ), sin(θ)) * sys.dt +
                        sys.σ_T * SVector(ξ[1], ξ[2])
        end
        angles[i + 1] = θ + sys.σ_R * ξ[3]
    end
    return xs, angles
end

@testset "pCN proposals and Gaussian-prior stationarity" begin
    bins = -100.0:1.0:100.0
    abp = ABP(Dt=0.025, Dr=0.025, v=0.31)
    for β in (0.3, 1.0)
        sys = ABPTrajectory(Xoshiro(8), abp, 0.2; x0=[-1.0, 0.0])
        alg = Multicanonical(Xoshiro(91), BinnedObject(bins, 0.0))
        expected_rng = deepcopy(alg.rng)
        noise_index = rand(expected_rng, eachindex(sys.ξs))
        expected_noise = sqrt(1 - β^2) * sys.ξs[noise_index] +
                         β * (@SVector randn(expected_rng, 3))
        ABPNoiseSweep.update!(sys, alg, endpoint_x; bins=bins, potential_active=false,
                              noise_proposal=:pcn, pcn_beta=β)
        @test sys.ξs[noise_index] == expected_noise
        reconstructed = copy(sys.xs)
        ABPNoiseSweep.integrate!(sys; potential_active=false)
        @test sys.xs == reconstructed

        sys = ABPTrajectory(Xoshiro(8), abp, 0.2; x0=[-1.0, 0.0])
        alg = Multicanonical(Xoshiro(91), BinnedObject(bins, 0.0))
        expected_rng = deepcopy(alg.rng)
        i0 = rand(expected_rng, 1:length(sys.ξs))
        i1 = min(length(sys.ξs), i0 + 6)
        expected_noises = copy(sys.ξs)
        for i in i0:i1
            expected_noises[i] = sqrt(1 - β^2) * sys.ξs[i] +
                                 β * (@SVector randn(expected_rng, 3))
        end
        ABPNoiseSweep.block_update!(sys, alg, endpoint_x; block_size=7, bins=bins,
                                    potential_active=false, noise_proposal=:pcn, pcn_beta=β)
        @test sys.ξs == expected_noises

        moves, _ = ABPNoiseSweep.make_abp_moves(endpoint_x, bins, false;
                                               noise_proposal=:pcn,
                                               block_pcn_beta=β, local_pcn_beta=β)
        for move in moves
            @test move(sys, alg) === sys
        end
    end

    # beta=1 with one noise vector draws independent samples from N(0,I).
    # A constant endpoint bias must leave this prior invariant, with no
    # additional Gaussian-prior Metropolis rejection.
    sys = ABPTrajectory(Xoshiro(8), abp, 0.01)
    alg = Multicanonical(Xoshiro(901), BinnedObject(bins, 0.0))
    samples = zeros(3, 4096)
    for k in axes(samples, 2)
        ABPNoiseSweep.update!(sys, alg, endpoint_x; bins=bins, potential_active=false,
                              noise_proposal=:pcn, pcn_beta=1.0)
        samples[:, k] .= sys.ξs[1]
    end
    for j in 1:3
        @test abs(sum(samples[j, :]) / size(samples, 2)) < 0.07
        @test 0.88 < sum(abs2, samples[j, :]) / size(samples, 2) < 1.12
    end

    # The proposal-density ratio cancels the Gaussian prior for general beta.
    x = SVector(0.8, -1.2, 0.4)
    y = SVector(-0.3, 0.7, 1.1)
    β = 0.4
    α = sqrt(1 - β^2)
    log_forward = -sum(abs2, y - α * x) / (2β^2)
    log_reverse = -sum(abs2, x - α * y) / (2β^2)
    @test -0.5 * (sum(abs2, y) - sum(abs2, x)) + log_reverse - log_forward ≈ 0 atol=1e-14

    for proposal in (:block, :local)
        rejecting_bins = 1.0:0.01:1.1
        sys = ABPTrajectory(Xoshiro(8), abp, 0.2; x0=[-1.0, 0.0])
        old_positions, old_angles, old_noises = copy(sys.xs), copy(sys.θs), copy(sys.ξs)
        alg = Multicanonical(Xoshiro(91), BinnedObject(rejecting_bins, 0.0))
        if proposal === :block
            ABPNoiseSweep.block_update!(sys, alg, endpoint_x; block_size=7, bins=rejecting_bins,
                                        potential_active=false, noise_proposal=:pcn, pcn_beta=0.5)
        else
            ABPNoiseSweep.update!(sys, alg, endpoint_x; bins=rejecting_bins,
                                  potential_active=false, noise_proposal=:pcn, pcn_beta=0.5)
        end
        @test sys.xs == old_positions
        @test sys.θs == old_angles
        @test sys.ξs == old_noises
    end
end

@testset "Euler default and deterministic Crank–Nicolson noise map" begin
    abp = ABP(Dt=0.025, Dr=0.025, v=0.31)
    for potential_active in (false, true)
        euler = ABPTrajectory(Xoshiro(2026), abp, 1.0;
                              x0=[-1.0, 0.0], potential_active=potential_active)
        expected_xs, expected_angles = reference_euler_positions(euler; potential_active=potential_active)
        @test euler.integrator === :euler
        @test euler.xs == expected_xs
        @test euler.θs == expected_angles

        cn = ABPTrajectory(Xoshiro(2026), abp, 1.0;
                           x0=[-1.0, 0.0], potential_active=potential_active,
                           integrator=:crank_nicolson)
        @test cn.ξs == euler.ξs
        @test cn.θs == euler.θs
        @test rand(cn.rng, 10) == rand(euler.rng, 10)
        for i in eachindex(cn.ξs)
            x, x_next = cn.xs[i], cn.xs[i + 1]
            θ, θ_next = cn.θs[i], cn.θs[i + 1]
            ξ = cn.ξs[i]
            expected_delta = cn.v * cn.dt / 2 * (
                SVector(cos(θ), sin(θ)) + SVector(cos(θ_next), sin(θ_next))
            ) + cn.σ_T * SVector(ξ[1], ξ[2])
            if potential_active
                expected_delta -= cn.dt / 2 * (
                    ABPNoiseSweep.gradient(x) + ABPNoiseSweep.gradient(x_next)
                )
            end
            @test norm(x_next - x - expected_delta, Inf) <= 5e-12
        end
        whole_path = copy(cn.xs)
        whole_angles = copy(cn.θs)
        ABPNoiseSweep.integrate!(cn, 25:length(cn.ξs); potential_active=potential_active)
        @test cn.xs == whole_path
        @test cn.θs == whole_angles
    end
    @test_throws ErrorException ABPTrajectory(Xoshiro(1), abp, 0.1; integrator=:invalid)
    @test_throws ErrorException ABPNoiseSweep.crank_nicolson_position(
        SVector(0.0, 0.0), SVector(NaN, 0.0), 0.005)
end

@testset "Potential derivatives for the implicit solve" begin
    abp = ABP()
    sys = ABPTrajectory(Xoshiro(4), abp, 0.01)
    δ = 1e-5
    for position in (SVector(-1.2, -0.7), SVector(0.2, 0.0), SVector(0.6, 1.1))
        hessian = ABPNoiseSweep.potential_hessian(position)
        for j in 1:2
            step = j == 1 ? SVector(δ, 0.0) : SVector(0.0, δ)
            sys.xs[1] = position + step
            u_plus = ABPNoiseSweep.potential(sys, 1)
            sys.xs[1] = position - step
            u_minus = ABPNoiseSweep.potential(sys, 1)
            @test ABPNoiseSweep.gradient(position)[j] ≈ (u_plus - u_minus) / (2δ) atol=1e-8
            finite_difference = (ABPNoiseSweep.gradient(position + step) -
                                 ABPNoiseSweep.gradient(position - step)) / (2δ)
            @test hessian[:, j] ≈ finite_difference atol=1e-8
        end
    end
end

# Original block proposal, deliberately retaining its allocations, as a seeded
# acceptance/rejection reference for the scratch-buffer optimization.
function reference_block_update!(sys, alg; block_size, bins, potential_active)
    noise_count = length(sys.ξs)
    i0 = rand(alg.rng, 1:noise_count)
    i1 = min(noise_count, i0 + min(block_size, noise_count) - 1)
    old_noise = copy(sys.ξs[i0:i1])
    new_noise = similar(old_noise)
    log_prior_ratio = 0.0
    for k in eachindex(old_noise)
        proposed = old_noise[k] + 0.05 * (@SVector randn(alg.rng, 3))
        new_noise[k] = proposed
        log_prior_ratio += -0.5 * (sum(proposed .^ 2) - sum(old_noise[k] .^ 2))
    end
    if rand(alg.rng) < exp(min(0.0, log_prior_ratio))
        sys.tmp_xs[i0:end] .= sys.xs[i0:end]
        sys.tmp_θs[i0:end] .= sys.θs[i0:end]
        old_endpoint = endpoint_x(sys)
        sys.ξs[i0:i1] .= new_noise
        ABPNoiseSweep.integrate!(sys, i0:noise_count; potential_active=potential_active)
        new_endpoint = endpoint_x(sys)
        outside = !ABPNoiseSweep.in_bin_range(old_endpoint, bins) ||
                  !ABPNoiseSweep.in_bin_range(new_endpoint, bins)
        if outside || !accept!(alg, new_endpoint, old_endpoint)
            sys.xs[i0:end] .= sys.tmp_xs[i0:end]
            sys.θs[i0:end] .= sys.tmp_θs[i0:end]
            sys.ξs[i0:i1] .= old_noise
        end
    end
    alg.steps += 1
    return sys
end

function reference_local_update!(sys, alg; bins, potential_active)
    noise_index = rand(alg.rng, eachindex(sys.ξs))
    old_noise = sys.ξs[noise_index]
    proposed = old_noise + 0.8 * (@SVector randn(alg.rng, 3))
    log_prior_ratio = -0.5 * (sum(proposed .^ 2) - sum(old_noise .^ 2))
    if rand(alg.rng) < exp(min(0.0, log_prior_ratio))
        sys.tmp_xs[noise_index:end] .= sys.xs[noise_index:end]
        sys.tmp_θs[noise_index:end] .= sys.θs[noise_index:end]
        old_endpoint = endpoint_x(sys)
        sys.ξs[noise_index] = proposed
        ABPNoiseSweep.integrate!(sys, noise_index:length(sys.ξs); potential_active=potential_active)
        new_endpoint = endpoint_x(sys)
        outside = !ABPNoiseSweep.in_bin_range(old_endpoint, bins) ||
                  !ABPNoiseSweep.in_bin_range(new_endpoint, bins)
        if outside || !accept!(alg, new_endpoint, old_endpoint)
            sys.xs[noise_index:end] .= sys.tmp_xs[noise_index:end]
            sys.θs[noise_index:end] .= sys.tmp_θs[noise_index:end]
            sys.ξs[noise_index] = old_noise
        end
    end
    alg.steps += 1
    return sys
end

@testset "Scratch-buffer proposals preserve seeded transitions" begin
    for integrator in (:euler, :crank_nicolson), bins in (-2.0:0.05:2.0, 1.0:0.01:1.1), move in (:block, :local)
        abp = ABP(Dt=0.025, Dr=0.025, v=0.31)
        original = ABPTrajectory(Xoshiro(8), abp, 0.2;
                                x0=[-1.0, 0.0], potential_active=true, integrator=integrator)
        optimized = deepcopy(original)
        alg_original = Multicanonical(Xoshiro(19), BinnedObject(bins, 0.0))
        alg_optimized = deepcopy(alg_original)
        for _ in 1:50
            if move === :block
                reference_block_update!(original, alg_original; block_size=7, bins=bins, potential_active=true)
                ABPNoiseSweep.block_update!(optimized, alg_optimized, endpoint_x;
                                            block_size=7, bins=bins, potential_active=true)
            else
                reference_local_update!(original, alg_original; bins=bins, potential_active=true)
                ABPNoiseSweep.update!(optimized, alg_optimized, endpoint_x;
                                     bins=bins, potential_active=true)
            end
            @test optimized.ξs == original.ξs
            @test optimized.xs == original.xs
            @test optimized.θs == original.θs
            @test alg_optimized.steps == alg_original.steps
        end
        @test rand(alg_optimized.rng, 10) == rand(alg_original.rng, 10)
    end
end

# Optional, bounded allocation measurement for the change report. This compares
# the same seeded sequence after warmup; it is not a machine-sensitive test.
function abp_measure_move_allocations(; trajectory_T=40.0, n_moves=100)
    bins = -10.0:0.05:10.0
    results = NamedTuple[]
    for move in (:block, :local)
        original = ABPTrajectory(Xoshiro(8), ABP(Dt=0.025, Dr=0.025, v=0.31), trajectory_T;
                                x0=[-1.0, 0.0], potential_active=true)
        optimized = deepcopy(original)
        alg_original = Multicanonical(Xoshiro(19), BinnedObject(bins, 0.0))
        alg_optimized = deepcopy(alg_original)
        reference_step! = move === :block ?
            () -> reference_block_update!(original, alg_original; block_size=100, bins=bins, potential_active=true) :
            () -> reference_local_update!(original, alg_original; bins=bins, potential_active=true)
        optimized_step! = move === :block ?
            () -> ABPNoiseSweep.block_update!(optimized, alg_optimized, endpoint_x;
                                              block_size=100, bins=bins, potential_active=true) :
            () -> ABPNoiseSweep.update!(optimized, alg_optimized, endpoint_x;
                                       bins=bins, potential_active=true)
        reference_step!()
        optimized_step!()
        GC.gc()
        before = @allocated for _ in 1:n_moves
            reference_step!()
        end
        GC.gc()
        after = @allocated for _ in 1:n_moves
            optimized_step!()
        end
        @assert original.xs == optimized.xs && original.ξs == optimized.ξs
        push!(results, (move=move, trajectory_T=trajectory_T, n_moves=n_moves,
                        original_bytes_per_move=before / n_moves,
                        optimized_bytes_per_move=after / n_moves))
    end
    return results
end
