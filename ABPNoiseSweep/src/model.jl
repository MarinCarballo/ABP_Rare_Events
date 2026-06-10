# ABP model, trajectory integration, potential, and observables.

# =============================================================================
# Notebook cell 7
# =============================================================================
#First define the ABP system parameters in a struct for easy access and modification.
struct ABP
    Dt::Float64
    v::Float64
    Dr::Float64

    function ABP(; Dt::Float64=0.01, v::Float64=0.36, Dr::Float64=0.01)
        new(Dt, v, Dr)
    end
end

mutable struct ABPTrajectory #mutable because we will update the trajectory in place during integration
    rng :: AbstractRNG
    dt  :: Float64
    T   :: Float64
    v   :: Float64
    Dt  :: Float64
    Dr  :: Float64
    σ_T :: Float64
    σ_R :: Float64

    ξs  :: Vector{SVector{3, Float64}}
    xs  :: Vector{SVector{2, Float64}}
    θs  :: Vector{Float64}

    tmp_xs :: Vector{SVector{2, Float64}}
    tmp_θs :: Vector{Float64}

    function ABPTrajectory(
        rng::AbstractRNG,
        abp::ABP,
        T::Float64;
        dt::Float64=1e-2,
        x0::Vector{Float64}=[0.0, 0.0],
        θ0::Symbol=:uniform,
        potential_active::Bool=false,
    )
        N   = round(Int, T / dt) + 1  # number of time steps
        σ_T = sqrt(2 * abp.Dt * dt) # translational noise strength
        σ_R = sqrt(2 * abp.Dr * dt) # rotational noise strength

        ξs = [@SVector randn(rng, 3) for _ in 1:N-1] # vector of noise in x, y, and θ for each time step
        xs = [@SVector zeros(2) for _ in 1:N] # vector of positions at each time step
        θs = zeros(N) # vector of orientations at each time step

        xs[1] = SVector{2, Float64}(x0) # set initial position

        if θ0 == :uniform # set initial orientation uniformly at random in [0, 2π)
            θs[1] = rand(rng, Uniform(0, 2π))
        elseif θ0 == :zero
            θs[1] = 0.0
        else
            error("Unsupported θ0 = $θ0")
        end

        tmp_xs = copy(xs) # temporary storage for positions during integration
        tmp_θs = copy(θs) # temporary storage for orientations during integration

        sys = new(rng, dt, T, abp.v, abp.Dt, abp.Dr, σ_T, σ_R, ξs, xs, θs, tmp_xs, tmp_θs,)

        integrate!(sys; potential_active=potential_active) # integrate the trajectory to fill in the positions and orientations

        sys.tmp_xs .= sys.xs # copy the integrated positions into the temporary storage (this will be used for MCMC updates)
        sys.tmp_θs .= sys.θs

        return sys
    end
end

# =============================================================================
# Notebook cell 9
# =============================================================================
g(x) = (x - 1.0)^2 * (x + 1.0)^2

function potential(sys::ABPTrajectory, i::Int)
    x = sys.xs[i][1]
    y = sys.xs[i][2]

    lambda_ = 2.5
    k = 0.3
    
    f_0   = 0.5 * lambda_ * y^2
    f_bar = 1.0 + 0.5 * k * y^2

    denom = f_0 + f_bar
    f_mix = denom == 0.0 ? 0.0 : (f_0 * f_bar) / denom

    return g(x) * (f_bar - f_mix) + f_mix
end

function gradient(sys::ABPTrajectory, i::Int)
    x = sys.xs[i][1]
    y = sys.xs[i][2]

    lambda_ = 2.5
    k = 0.3

    gxy   = (x - 1.0)^2 * (x + 1.0)^2
    dg_dx = 4.0 * x * (x^2 - 1.0)

    f_0   = 0.5 * lambda_ * y^2
    f_bar = 1.0 + 0.5 * k * y^2

    denom = f_0 + f_bar
    f_mix = denom == 0.0 ? 0.0 : (f_0 * f_bar) / denom

    dfbar_dy = k * y

    dfmix_dy =
        denom == 0.0 ? 0.0 :
        y * (lambda_ * f_bar^2 + k * f_0^2) / denom^2

    df_dx = dg_dx * (f_bar - f_mix)
    df_dy = gxy * (dfbar_dy - dfmix_dy) + dfmix_dy

    return SVector(df_dx, df_dy)
end

# =============================================================================
# Notebook cell 11
# =============================================================================
function integrate!(sys::ABPTrajectory, r::UnitRange{Int}=1:0; potential_active::Bool=false,)
    r = isempty(r) ? eachindex(sys.ξs) : r

    θ = sys.θs[first(r)]
    x = sys.xs[first(r)]

    if potential_active
        for time_index in r
            ξ = sys.ξs[time_index] # take the noise for this time step
            grad = gradient(sys, time_index) # compute the gradient of the potential at the current position

            x = x +
                sys.v * SVector(cos(θ), sin(θ)) * sys.dt + 
                sys.σ_T * SVector(ξ[1], ξ[2]) -
                grad * sys.dt  #euler-maruyama update with potential gradient

            θ = θ + sys.σ_R * ξ[3]

            sys.xs[time_index + 1] = x
            sys.θs[time_index + 1] = θ
        end
    else
        for time_index in r
            ξ = sys.ξs[time_index]

            x = x +
                sys.v * SVector(cos(θ), sin(θ)) * sys.dt +
                sys.σ_T * SVector(ξ[1], ξ[2])

            θ = θ + sys.σ_R * ξ[3]

            sys.xs[time_index + 1] = x
            sys.θs[time_index + 1] = θ
        end
    end

    return sys
end

# =============================================================================
# Notebook cell 13
# =============================================================================
endpoint_x(sys::ABPTrajectory) = sys.xs[end][1]
endpoint_y(sys::ABPTrajectory) = sys.xs[end][2]
endpoint_theta(sys::ABPTrajectory) = mod2pi(sys.θs[end])
endpoint_distance(sys::ABPTrajectory) = norm(sys.xs[end])
path_y_int(sys::ABPTrajectory) = sys.dt * sum(x[2] for x in sys.xs)
mean_y(sys::ABPTrajectory) = mean(x[2] for x in sys.xs)
max_x(sys::ABPTrajectory) = maximum(x[1] for x in sys.xs)

# =============================================================================
# Notebook cell 15
# =============================================================================
const right_center = SVector(1.0, 0.0)
const right_radius = 0.2

const left_center = SVector(-1.0, 0.0)
const left_radius = 0.2

function entered_right_well(xs; center=right_center, radius=right_radius)
    any(norm(x - center) <= radius for x in xs)
end

function ended_in_right_well(xs; center=right_center, radius=right_radius)
    norm(xs[end] - center) <= radius
end

entered_right_well(sys::ABPTrajectory) = entered_right_well(sys.xs)
ended_in_right_well(sys::ABPTrajectory) = ended_in_right_well(sys.xs)

# =============================================================================
