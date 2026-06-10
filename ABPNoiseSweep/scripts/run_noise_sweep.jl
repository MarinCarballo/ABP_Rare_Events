#!/usr/bin/env julia

# Non-interactive entry point for local or cluster execution.
# Use from repository root:
#   julia --project=. scripts/run_noise_sweep.jl --smoke
#
# Dependency setup is controlled by environment variables:
#   ABP_INSTANTIATE=true   -> run Pkg.instantiate() before loading the package
#   ABP_PRECOMPILE=true    -> run Pkg.precompile() before loading the package
# Keep both false for normal production jobs after the environment is installed.

import Pkg

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
Pkg.activate(PROJECT_ROOT)

_truthy(x) = lowercase(strip(x)) in ("1", "true", "yes", "y", "on")

if _truthy(get(ENV, "ABP_INSTANTIATE", "false"))
    @info "Running Pkg.instantiate()" project=PROJECT_ROOT
    Pkg.instantiate()
end

if _truthy(get(ENV, "ABP_PRECOMPILE", "false"))
    @info "Running Pkg.precompile()" project=PROJECT_ROOT
    Pkg.precompile()
end

using ABPNoiseSweep

ABPNoiseSweep.main(ARGS)
