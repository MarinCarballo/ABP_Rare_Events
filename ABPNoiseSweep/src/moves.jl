# Path-space MCMC proposal moves.

function in_bin_range(x, bins) # check if x is within the range defined by the first and last edges of the bins
    first(bins) <= x < last(bins)
end

function update!(
    sys::ABPTrajectory,
    alg::ImportanceSampling{<:MulticanonicalEnsemble},
    obs::Function,
    Δξ::Float64=0.8;
    bins=nothing,
    potential_active::Bool=true,
)
    noise_index = rand(alg.rng, eachindex(sys.ξs))

    old_ξ = sys.ξs[noise_index]
    new_ξ = old_ξ + Δξ * (@SVector randn(alg.rng, 3))

    log_prior_ratio = -0.5 * (sum(new_ξ .^ 2) - sum(old_ξ .^ 2))

    if rand(alg.rng) < exp(min(0.0, log_prior_ratio))
        sys.tmp_xs[noise_index:end] .= sys.xs[noise_index:end]
        sys.tmp_θs[noise_index:end] .= sys.θs[noise_index:end]

        d_old = obs(sys)

        sys.ξs[noise_index] = new_ξ

        integrate!(
            sys,
            noise_index:length(sys.ξs);
            potential_active=potential_active,
        )

        d_new = obs(sys)

        outside_bins = # reject if the proposed trajectory falls outside the bin range of the observable (if bins are defined)
            bins !== nothing &&
            (!in_bin_range(d_old, bins) || !in_bin_range(d_new, bins))

        if outside_bins || !accept!(alg, d_new, d_old)
            sys.xs[noise_index:end] .= sys.tmp_xs[noise_index:end]
            sys.θs[noise_index:end] .= sys.tmp_θs[noise_index:end]
            sys.ξs[noise_index] = old_ξ
        end
    end

    alg.steps += 1

    return sys
end

# =============================================================================
# Notebook cell 20
# =============================================================================
function block_update!(
    sys::ABPTrajectory,
    alg::ImportanceSampling{<:MulticanonicalEnsemble},
    obs::Function,
    Δξ::Float64=0.05;
    block_size::Int=100,
    bins=nothing,
    potential_active::Bool=true,
)
    Nξ = length(sys.ξs)

    block_size = min(block_size, Nξ) # ensure block size does not exceed number of noise variables

    i0 = rand(alg.rng, 1:Nξ) # randomly select the starting index of the block
    i1 = min(Nξ, i0 + block_size - 1) # randomly select ending index

    old_ξs = copy(sys.ξs[i0:i1]) 
    new_ξs = similar(old_ξs)

    log_prior_ratio = 0.0

    for k in eachindex(old_ξs) # loop over the selected block of noise variables
        new_ξ = old_ξs[k] + Δξ * (@SVector randn(alg.rng, 3))
        new_ξs[k] = new_ξ

        log_prior_ratio += -0.5 * (sum(new_ξ .^ 2) - sum(old_ξs[k] .^ 2))
    end

    if rand(alg.rng) < exp(min(0.0, log_prior_ratio))
        sys.tmp_xs[i0:end] .= sys.xs[i0:end]
        sys.tmp_θs[i0:end] .= sys.θs[i0:end]

        d_old = obs(sys)

        sys.ξs[i0:i1] .= new_ξs

        integrate!(
            sys,
            i0:length(sys.ξs);
            potential_active=potential_active,
        )

        d_new = obs(sys)

        outside_bins =
            bins !== nothing &&
            (!in_bin_range(d_old, bins) || !in_bin_range(d_new, bins))

        if outside_bins || !accept!(alg, d_new, d_old)
            sys.xs[i0:end] .= sys.tmp_xs[i0:end]
            sys.θs[i0:end] .= sys.tmp_θs[i0:end]
            sys.ξs[i0:i1] .= old_ξs
        end
    end

    alg.steps += 1

    return sys
end

# =============================================================================
# Notebook cell 22
# =============================================================================
#only updates initial orientation for each MCMC step.
function theta0_update!(
    sys::ABPTrajectory,
    alg::ImportanceSampling{<:MulticanonicalEnsemble},
    obs::Function;
    bins=nothing,
    potential_active::Bool=true,
)
    Nξ = length(sys.ξs)

    old_θ0 = sys.θs[1]

    sys.tmp_xs .= sys.xs
    sys.tmp_θs .= sys.θs

    d_old = obs(sys)

    sys.θs[1] = rand(alg.rng, Uniform(0, 2π))

    integrate!(
        sys,
        1:Nξ;
        potential_active=potential_active,
    )

    d_new = obs(sys)

    outside_bins =
        bins !== nothing &&
        (!in_bin_range(d_old, bins) || !in_bin_range(d_new, bins))

    if outside_bins || !accept!(alg, d_new, d_old)
        sys.θs[1] = old_θ0
        sys.xs .= sys.tmp_xs
        sys.θs .= sys.tmp_θs
    end

    alg.steps += 1

    return sys
end

# =============================================================================
# Notebook cell 24
# =============================================================================
function reflect_y!(sys::ABPTrajectory)
    for i in eachindex(sys.xs)
        sys.xs[i] = SVector(sys.xs[i][1], -sys.xs[i][2])
        sys.θs[i] = 2π - sys.θs[i]
    end

    for i in eachindex(sys.ξs)
        sys.ξs[i] = SVector(sys.ξs[i][1], -sys.ξs[i][2], -sys.ξs[i][3])
    end

    sys.tmp_xs .= sys.xs
    sys.tmp_θs .= sys.θs

    return sys
end

function reflection_update!(
    sys::ABPTrajectory,
    alg::ImportanceSampling{<:MulticanonicalEnsemble},
    obs::Function;
    bins=nothing,
)
    d_old = obs(sys)

    reflect_y!(sys)

    d_new = obs(sys)

    outside_bins =
        bins !== nothing &&
        (!in_bin_range(d_old, bins) || !in_bin_range(d_new, bins))

    if outside_bins || !accept!(alg, d_new, d_old)
        reflect_y!(sys)
    end

    alg.steps += 1

    return sys
end

# =============================================================================
# Notebook cell 26
# =============================================================================
function make_abp_moves(
    obs,
    bins_bias,
    potential_active;
    move_weights::AbstractVector{<:Real}=Float64[0.0, 0.05, 0.55, 0.40],
    block_dxi::Real=0.05,
    local_dxi::Real=0.8,
)
    @assert length(move_weights) == 4 "move_weights must be [reflection, theta0, block, local]."
    @assert all(w -> w >= 0.0, move_weights) "move_weights must be nonnegative."
    @assert sum(move_weights) > 0.0 "At least one move weight must be positive."
    @assert block_dxi > 0.0 "block_dxi must be positive."
    @assert local_dxi > 0.0 "local_dxi must be positive."

    block_dxi = Float64(block_dxi)
    local_dxi = Float64(local_dxi)

    moves = [
        (sys, alg) -> reflection_update!(
            sys,
            alg,
            obs;
            bins=bins_bias,
        ),

        (sys, alg) -> theta0_update!(
            sys,
            alg,
            obs;
            bins=bins_bias,
            potential_active=potential_active,
        ),

        (sys, alg) -> block_update!(
            sys,
            alg,
            obs,
            block_dxi;
            block_size=rand(alg.rng, [50, 100, 250, 500, 1000]),
            bins=bins_bias,
            potential_active=potential_active,
        ),

        (sys, alg) -> update!(
            sys,
            alg,
            obs,
            local_dxi;
            bins=bins_bias,
            potential_active=potential_active,
        ),
    ]

    return moves, Weights(Float64.(move_weights))
end

function random_move!(sys, alg, moves, move_weights)
    move = sample(alg.rng, moves, move_weights)
    return move(sys, alg)
end

# -----------------------------
