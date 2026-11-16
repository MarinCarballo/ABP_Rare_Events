using Test, Random, StaticArrays

function component_test_trajectory(positions, angles; v=0.31, dt=0.01)
    @assert length(positions) == length(angles)
    sys = ABPTrajectory(Xoshiro(812), ABP(Dt=0.025, Dr=0.025, v=v),
                        dt * (length(positions) - 1); dt=dt)
    sys.xs .= positions
    sys.θs .= angles
    return sys
end

function component_test_accumulator()
    ex = [-2.0, 0.0, 2.0]
    ey = [-4.0, 0.0, 4.0]
    et = [0.0, π, 2π]
    acc = ABPNoiseSweep.abp_make_prod_accumulator(ex, ey, ey, ey, ex, ey, et, 1, 1)
    return acc, ex, ey, et
end

# Literal previous expression and reduction order: the existing total must not
# be replaced by the sum of separately rounded Cartesian component totals.
function component_legacy_total(sys)
    W_active = 0.0
    @inbounds for k in eachindex(sys.ξs)
        r0 = sys.xs[k]
        r1 = sys.xs[k + 1]
        dr = r1 - r0
        theta_mid = 0.5 * (sys.θs[k] + sys.θs[k + 1])
        W_active += sys.v * (
            cos(theta_mid) * dr[1] + sin(theta_mid) * dr[2]
        )
    end
    return W_active
end

@testset "Full-path Cartesian active-work definitions" begin
    θ = π / 6
    positions = [SVector(-1.0, 1.0), SVector(-0.2, 3.0),
                 SVector(0.7, -2.0), SVector(1.0, -1.0)]
    sys = component_test_trajectory(positions, fill(θ, 4))
    rng_before = deepcopy(sys.rng)
    xs_before, θs_before, ξs_before = copy(sys.xs), copy(sys.θs), copy(sys.ξs)
    metrics = active_work_and_first_lr_crossing(sys)
    @test metrics.active_work_x ≈ sys.v * cos(θ) * (positions[end][1] - positions[1][1])
    @test metrics.active_work_y ≈ sys.v * sin(θ) * (positions[end][2] - positions[1][2])
    @test metrics.active_work ≈ metrics.active_work_x + metrics.active_work_y
    components = ABPNoiseSweep.active_work_components(sys)
    @test components == (active_work=metrics.active_work,
                         active_work_x=metrics.active_work_x, active_work_y=metrics.active_work_y)
    @test ABPNoiseSweep.active_work_x(sys) == metrics.active_work_x
    @test ABPNoiseSweep.active_work_y(sys) == metrics.active_work_y
    @test sys.xs == xs_before && sys.θs == θs_before && sys.ξs == ξs_before
    @test rand(sys.rng, 8) == rand(rng_before, 8)

    # A closed y displacement can carry nonzero y work when the orientation
    # changes: v integral sin(theta) dy is not v sin(theta_final) Delta_y.
    closed_y = component_test_trajectory(
        [SVector(-1.0, 0.0), SVector(0.5, 1.0), SVector(1.0, 0.0)], [0.0, 0.0, π],
    )
    result = active_work_and_first_lr_crossing(closed_y)
    @test closed_y.xs[end][2] == closed_y.xs[1][2]
    @test result.active_work_y ≈ -closed_y.v
    @test abs(result.active_work_y) > 0

    # First crossing labels the complete path; its work includes all later
    # increments even if all y motion occurs after the first crossing.
    after_crossing = component_test_trajectory(
        [SVector(-1.0, 0.0), SVector(0.0, 0.0), SVector(0.5, 2.0), SVector(1.0, 3.0)],
        fill(π / 2, 4),
    )
    after = active_work_and_first_lr_crossing(after_crossing)
    @test after.has_crossing
    @test after.crossing_time == after_crossing.dt
    @test after.crossing_y == 0.0
    @test after.active_work_y ≈ 3 * after_crossing.v
    @test after.active_work ≈ 3 * after_crossing.v
end

@testset "Legacy total bits and fused-scan component equivalence" begin
    for integrator in (:euler, :crank_nicolson), seed in 1:8
        sys = ABPTrajectory(Xoshiro(seed), ABP(Dt=0.025, Dr=0.025, v=0.31), 2.0;
                            x0=[-1.0, 0.0], potential_active=true, integrator=integrator)
        legacy = component_legacy_total(sys)
        metrics = active_work_and_first_lr_crossing(sys)
        @test reinterpret(UInt64, metrics.active_work) == reinterpret(UInt64, legacy)
        @test reinterpret(UInt64, active_work(sys)) == reinterpret(UInt64, legacy)
        @test metrics.active_work ≈ metrics.active_work_x + metrics.active_work_y atol=1e-14
        acc, ex, ey, et = component_test_accumulator()
        # Every occupation point is filtered out, and time stride is coarse;
        # neither setting can thin or filter the work integrals.
        fused = ABPNoiseSweep.abp_add_whole_path_conditioned_windows!(
            acc, sys, [1], ex, ey, et, 1, 2.0;
            path_time_stride=17, path_x_min=100.0, compute_route_metrics=true,
        )
        @test isequal(fused, metrics)
        @test reinterpret(UInt64, fused.active_work) == reinterpret(UInt64, legacy)
        @test sum(acc.counts_path_xy_biased[1]) == 0.0
        @test sum(acc.counts_path_xtheta_biased) == 0.0
    end
end

function component_test_metrics(wx, wy; crossing_y=1.0, has_crossing=true)
    return (active_work=wx + wy, active_work_x=wx, active_work_y=wy,
            crossing_y=crossing_y, crossing_time=0.02, has_crossing=has_crossing)
end

@testset "Component sufficient statistics share weights, selection, and covariance" begin
    acc, _, ey, _ = component_test_accumulator()
    observations = [(component_test_metrics(2.0, -1.0), 0.5),
                    (component_test_metrics(4.0, 3.0), 2.0),
                    (component_test_metrics(-3.0, 2.0), 0.0)]
    for (metrics, weight) in observations
        @test ABPNoiseSweep.abp_add_active_work_crossing!(acc, ey, metrics, weight)
    end
    @test acc.active_work_mass_biased == [0.0, 3.0]
    @test acc.active_work_mass_unbiased == [0.0, 2.5]
    @test acc.active_work_mass2_unbiased == [0.0, 4.25]
    @test acc.active_work_x_sum_biased == [0.0, 3.0]
    @test acc.active_work_y_sum_biased == [0.0, 4.0]
    @test acc.active_work_x_sum2_biased == [0.0, 29.0]
    @test acc.active_work_y_sum2_biased == [0.0, 14.0]
    @test acc.active_work_xy_sum_biased == [0.0, 4.0]
    @test acc.active_work_x_sum_unbiased == [0.0, 9.0]
    @test acc.active_work_y_sum_unbiased == [0.0, 5.5]
    @test acc.active_work_x_sum2_unbiased == [0.0, 34.0]
    @test acc.active_work_y_sum2_unbiased == [0.0, 18.5]
    @test acc.active_work_xy_sum_unbiased == [0.0, 23.0]
    # Retaining the cross moment reconstructs the total second moment and
    # variance. Adding x/y variances without covariance would be incorrect.
    mass = acc.active_work_mass_unbiased[2]
    mx, my = acc.active_work_x_sum_unbiased[2] / mass, acc.active_work_y_sum_unbiased[2] / mass
    vx = acc.active_work_x_sum2_unbiased[2] / mass - mx^2
    vy = acc.active_work_y_sum2_unbiased[2] / mass - my^2
    covariance = acc.active_work_xy_sum_unbiased[2] / mass - mx * my
    total_var = acc.active_work_sum2_unbiased[2] / mass - (acc.active_work_sum_unbiased[2] / mass)^2
    @test total_var ≈ vx + vy + 2 * covariance
    @test acc.active_work_sum2_unbiased[2] ≈ acc.active_work_x_sum2_unbiased[2] +
          acc.active_work_y_sum2_unbiased[2] + 2 * acc.active_work_xy_sum_unbiased[2]

    # Missing crossings and crossing ordinates outside the histogram must not
    # create component observations or alter the existing total moments.
    before = deepcopy(acc)
    @test !ABPNoiseSweep.abp_add_active_work_crossing!(
        acc, ey, component_test_metrics(100.0, 200.0; has_crossing=false), 3.0)
    @test !ABPNoiseSweep.abp_add_active_work_crossing!(
        acc, ey, component_test_metrics(100.0, 200.0; crossing_y=10.0), 3.0)
    for field in propertynames(acc)
        startswith(string(field), "active_work_") || continue
        @test getproperty(acc, field) == getproperty(before, field)
    end
    @test acc.n_active_work_observations[1] == 5
    @test acc.n_active_work_no_crossing[1] == 1
    @test acc.n_active_work_crossing_y_out[1] == 1

    # Per-chain accumulation followed by the existing merger must produce the
    # same component sufficient statistics as accumulating pooled observations.
    first_chain, _, _, _ = component_test_accumulator()
    second_chain, _, _, _ = component_test_accumulator()
    pooled, _, _, _ = component_test_accumulator()
    for (i, (metrics, weight)) in enumerate(observations)
        ABPNoiseSweep.abp_add_active_work_crossing!(i == 1 ? first_chain : second_chain, ey, metrics, weight)
        ABPNoiseSweep.abp_add_active_work_crossing!(pooled, ey, metrics, weight)
    end
    ABPNoiseSweep.abp_merge_prod_accumulators!(first_chain, second_chain; max_saved_paths_per_window=0)
    for field in propertynames(pooled)
        startswith(string(field), "active_work_") || continue
        @test getproperty(first_chain, field) == getproperty(pooled, field)
    end
end

# Optional measurement for the review report, separate from machine-sensitive
# assertions. Root can call this after including the tests.
function component_measurement_allocations()
    sys = ABPTrajectory(Xoshiro(18), ABP(v=0.31), 40.0; x0=[-1.0, 0.0], potential_active=true)
    acc, ex, ey, et = component_test_accumulator()
    windows = [1]
    scan!() = ABPNoiseSweep.abp_add_whole_path_conditioned_windows!(
        acc, sys, windows, ex, ey, et, 1, 2.0;
        path_time_stride=5, path_x_min=-1.0, compute_route_metrics=true,
    )
    active_work_and_first_lr_crossing(sys)
    scan!()
    return (standalone_bytes=@allocated(active_work_and_first_lr_crossing(sys)),
            fused_scan_bytes=@allocated(scan!()))
end
