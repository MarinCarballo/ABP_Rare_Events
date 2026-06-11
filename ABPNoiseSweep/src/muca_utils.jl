# MUCA extension and roundtrip diagnostic helpers.

function extend_muca_sides_tail_temperature!(
    alg::ImportanceSampling{<:MulticanonicalEnsemble};
    xT_min_extend::Real,
    xT_max_extend::Real,
    tail_temperature_low::Real,
    tail_temperature_high::Real,
    extend_low::Bool=true,
    extend_high::Bool=true,
)
    ens = ensemble(alg)

    if extend_high
        extend!(ens, :high; anchor=xT_max_extend, slope=-1 / tail_temperature_high)
    end

    if extend_low
        extend!(ens, :low; anchor=xT_min_extend, slope=-1 / tail_temperature_low)
    end

    return alg
end

# For the requested schedule: D=0.1 uses slope temperature 1; D=0.01 and 0.005 or higher use 2.
# A tail_temperature of 2 gives slope -1/2 = -0.5.
abp_tail_temperature_for_D(D::Real) = D >= 0.05 ? 1.0 : 2.0


function abp_roundtrip_steps_summary(
    iter_steps_per_target_roundtrips::AbstractVector{<:Real};
    window::Int,
    rtol::Real,
)
    finite_values = Float64[x for x in iter_steps_per_target_roundtrips if isfinite(x) && x > 0]
    if isempty(finite_values)
        return (
            steps_estimate = Inf,
            converged = false,
            rel_half_range = Inf,
            window_n = 0,
            note = "No finite per-target roundtrip estimate was available.",
        )
    end

    n = min(window, length(finite_values))
    tail = finite_values[end-n+1:end]
    med = median(tail)
    rel_half_range = med > 0 ? (maximum(tail) - minimum(tail)) / (2 * med) : Inf
    converged = n == window && rel_half_range <= rtol

    return (
        steps_estimate = med,
        converged = converged,
        rel_half_range = rel_half_range,
        window_n = n,
        note = converged ? "Tail-window estimate is stable within tolerance." : "Tail-window estimate has not reached the requested stability tolerance.",
    )
end

function abp_roundtrip_average_stop_summary(
    iter_avg_roundtrips_per_chain::AbstractVector{<:Real};
    target_avg_roundtrips_per_chain::Real,
    hits_required::Int,
)
    finite_values = Float64[x for x in iter_avg_roundtrips_per_chain if isfinite(x)]
    if isempty(finite_values)
        return (
            stop_iteration = 0,
            stopped_early = false,
            consecutive_hits = 0,
            last_avg_roundtrips_per_chain = NaN,
            target_avg_roundtrips_per_chain = Float64(target_avg_roundtrips_per_chain),
            note = "No finite average roundtrip values were available.",
        )
    end

    last_value = finite_values[end]
    hits = 0
    stop_iteration = 0
    for (i, value) in pairs(iter_avg_roundtrips_per_chain)
        if isfinite(value) && value >= target_avg_roundtrips_per_chain
            hits += 1
            if hits >= hits_required && stop_iteration == 0
                stop_iteration = i
            end
        else
            hits = 0
        end
    end

    stopped_early = stop_iteration > 0
    return (
        stop_iteration = stop_iteration,
        stopped_early = stopped_early,
        consecutive_hits = min(hits, hits_required),
        last_avg_roundtrips_per_chain = last_value,
        target_avg_roundtrips_per_chain = Float64(target_avg_roundtrips_per_chain),
        note = stopped_early ? "Average roundtrip target was met for the requested number of consecutive iterations." : "Average roundtrip target was not met for long enough to stop early.",
    )
end
